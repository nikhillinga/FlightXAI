"""Preprocessing module for FlightXAI.

Loads BTS flight data, cleans and samples records, splits into train/test
sets, engineers tabular features, and exports processed datasets and
feature artifacts.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    READABLE_COLUMNS,
    SAMPLE_SIZE,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_MONTHS,
    TRAIN_DATA_PATH,
    TRAIN_MONTHS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_holiday_window_dates(year: int = 2024) -> set:
    """Compute all dates within 2 days of key US public holidays.

    Key holidays: New Year, Memorial Day, Independence Day, Labor Day,
    Thanksgiving, Christmas.
    """
    holidays = [
        pd.Timestamp(f"{year-1}-12-25"),  # Christmas (prior year)
        pd.Timestamp(f"{year}-01-01"),    # New Year's Day
        pd.date_range(f"{year}-05-01", f"{year}-05-31", freq="W-MON")[-1],  # Memorial Day (last Mon of May)
        pd.Timestamp(f"{year}-07-04"),    # Independence Day
        pd.date_range(f"{year}-09-01", f"{year}-09-30", freq="W-MON")[0],   # Labor Day (first Mon of Sept)
        pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="W-THU")[3],   # Thanksgiving (4th Thu of Nov)
        pd.Timestamp(f"{year}-12-25"),    # Christmas Day
        pd.Timestamp(f"{year+1}-01-01"),  # New Year's Day (next year)
    ]
    window = set()
    for h in holidays:
        for offset in range(-2, 3):  # -2, -1, 0, 1, 2 days
            window.add((h + pd.Timedelta(days=offset)).date())
    return window


def _compute_holiday_flag(df: pd.DataFrame, holiday_window: set) -> pd.Series:
    """Determine whether each flight date is within 2 days of a holiday."""
    if {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(df.columns):
        dates = pd.to_datetime(
            {"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY_OF_MONTH"]},
            errors="coerce",
        )
    elif "FL_DATE" in df.columns:
        dates = pd.to_datetime(df["FL_DATE"], errors="coerce")
    else:
        dates = pd.to_datetime(
            {"year": 2024, "month": df["MONTH"], "day": df.get("DAY_OF_MONTH", 1)},
            errors="coerce",
        )
    return dates.dt.date.isin(holiday_window).astype(int)


def load_and_prepare(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BTS raw CSV files, preprocess, engineer features, and save datasets.

    Args:
        raw_dir: Path to directory containing raw BTS CSV files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    # 1. Glob all 2024 CSVs from raw_dir (skip 2025 files)
    pattern = os.path.join(raw_dir, "*2024*.csv")
    csv_files = sorted(glob.glob(pattern))
    csv_files = [f for f in csv_files if "2025" not in os.path.basename(f)]

    if not csv_files:
        raise FileNotFoundError(f"No 2024 CSV files found in {raw_dir}")

    logger.info("Found %d CSV files for 2024 in %s", len(csv_files), raw_dir)

    # 2. Read with memory-optimised dtypes
    dtypes = {
        "OP_UNIQUE_CARRIER": "category",
        "ORIGIN": "category",
        "DEST": "category",
        "DEP_DEL15": "float32",
        "CANCELLED": "float32",
        "DISTANCE": "float32",
        "YEAR": "int16",
        "MONTH": "int8",
        "DAY_OF_MONTH": "int8",
        "DAY_OF_WEEK": "int8",
    }

    needed_cols = {
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "FL_DATE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "DEST",
        "DEST_CITY_NAME",
        "CRS_DEP_TIME",
        "DEP_DELAY",
        "DEP_DEL15",
        "CANCELLED",
        "DISTANCE",
    }

    # 3. Concatenate all 2024 files
    dfs = []
    for f in csv_files:
        df_month = pd.read_csv(
            f,
            dtype={k: v for k, v in dtypes.items()},
            usecols=lambda c: c in needed_cols,
            low_memory=False,
        )
        dfs.append(df_month)

    df = pd.concat(dfs, ignore_index=True)
    total_raw_rows = len(df)
    logger.info("Total raw rows loaded: %d", total_raw_rows)

    # 4. Drop cancelled flights (CANCELLED == 1)
    df = df[df["CANCELLED"] != 1].copy()
    total_non_cancelled = len(df)
    logger.info(
        "Total non-cancelled rows: %d (dropped %d cancelled)",
        total_non_cancelled,
        total_raw_rows - total_non_cancelled,
    )

    # 5. Remap CRS_DEP_TIME == 2400 to 0 (BTS midnight encoding quirk)
    df.loc[df["CRS_DEP_TIME"] == 2400, "CRS_DEP_TIME"] = 0

    # 6. Sample SAMPLE_SIZE rows using RANDOM_STATE BEFORE splitting
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info("Sample size selected: %d", len(df))

    # 7. Create target: delay = DEP_DEL15.fillna(0).astype(int)
    df[TARGET_COLUMN] = df["DEP_DEL15"].fillna(0).astype(int)

    # 8. Split: train = MONTH in TRAIN_MONTHS, test = MONTH in TEST_MONTHS
    train_df = df[df["MONTH"].isin(TRAIN_MONTHS)].copy().reset_index(drop=True)
    test_df = df[df["MONTH"].isin(TEST_MONTHS)].copy().reset_index(drop=True)

    # 9. Engineer features — FIT on train only, TRANSFORM both:
    # departure_hour: CRS_DEP_TIME // 100
    train_df["departure_hour"] = (train_df["CRS_DEP_TIME"].fillna(0) // 100).astype(int)
    test_df["departure_hour"] = (test_df["CRS_DEP_TIME"].fillna(0) // 100).astype(int)

    # day_of_week: DAY_OF_WEEK
    train_df["day_of_week"] = train_df["DAY_OF_WEEK"].astype(int)
    test_df["day_of_week"] = test_df["DAY_OF_WEEK"].astype(int)

    # month: MONTH
    train_df["month"] = train_df["MONTH"].astype(int)
    test_df["month"] = test_df["MONTH"].astype(int)

    # is_weekend: 1 if DAY_OF_WEEK >= 6 else 0
    train_df["is_weekend"] = (train_df["day_of_week"] >= 6).astype(int)
    test_df["is_weekend"] = (test_df["day_of_week"] >= 6).astype(int)

    # distance: DISTANCE (StandardScaler — fit on train)
    scaler = StandardScaler()
    train_df["distance"] = scaler.fit_transform(train_df[["DISTANCE"]].fillna(0).astype("float32")).ravel()
    test_df["distance"] = scaler.transform(test_df[["DISTANCE"]].fillna(0).astype("float32")).ravel()

    # route_frequency: count per ORIGIN+DEST in train (fill unknowns in test with median)
    train_routes = train_df["ORIGIN"].astype(str) + "_" + train_df["DEST"].astype(str)
    test_routes = test_df["ORIGIN"].astype(str) + "_" + test_df["DEST"].astype(str)

    route_frequency_map = train_routes.value_counts().to_dict()
    median_route_freq = float(train_routes.value_counts().median()) if len(train_routes) > 0 else 0.0

    train_df["route_frequency"] = train_routes.map(route_frequency_map).fillna(median_route_freq).astype("float32")
    test_df["route_frequency"] = test_routes.map(route_frequency_map).fillna(median_route_freq).astype("float32")

    # Global means for fallback
    global_mean_delay = (
        float(train_df["DEP_DELAY"].mean())
        if len(train_df) > 0 and train_df["DEP_DELAY"].notna().any()
        else 0.0
    )
    global_delay_rate = (
        float(train_df[TARGET_COLUMN].mean())
        if len(train_df) > 0
        else 0.0
    )

    # carrier_avg_delay: mean DEP_DELAY per OP_UNIQUE_CARRIER in train (fill unknowns with global mean)
    carrier_avg_delay_map = train_df.groupby("OP_UNIQUE_CARRIER", observed=False)["DEP_DELAY"].mean().to_dict()
    train_df["carrier_avg_delay"] = train_df["OP_UNIQUE_CARRIER"].astype(str).map(carrier_avg_delay_map).fillna(global_mean_delay).astype("float32")
    test_df["carrier_avg_delay"] = test_df["OP_UNIQUE_CARRIER"].astype(str).map(carrier_avg_delay_map).fillna(global_mean_delay).astype("float32")

    # origin_avg_delay: mean DEP_DELAY per ORIGIN in train
    origin_avg_delay_map = train_df.groupby("ORIGIN", observed=False)["DEP_DELAY"].mean().to_dict()
    train_df["origin_avg_delay"] = train_df["ORIGIN"].astype(str).map(origin_avg_delay_map).fillna(global_mean_delay).astype("float32")
    test_df["origin_avg_delay"] = test_df["ORIGIN"].astype(str).map(origin_avg_delay_map).fillna(global_mean_delay).astype("float32")

    # dest_avg_delay: mean DEP_DELAY per DEST in train
    dest_avg_delay_map = train_df.groupby("DEST", observed=False)["DEP_DELAY"].mean().to_dict()
    train_df["dest_avg_delay"] = train_df["DEST"].astype(str).map(dest_avg_delay_map).fillna(global_mean_delay).astype("float32")
    test_df["dest_avg_delay"] = test_df["DEST"].astype(str).map(dest_avg_delay_map).fillna(global_mean_delay).astype("float32")

    # departure_peak_hour_flag: 1 if hour in [7,8,17,18,19]
    peak_hours = {7, 8, 17, 18, 19}
    train_df["departure_peak_hour_flag"] = train_df["departure_hour"].isin(peak_hours).astype(int)
    test_df["departure_peak_hour_flag"] = test_df["departure_hour"].isin(peak_hours).astype(int)

    # holiday_flag: 1 if within 2 days of US public holidays
    holiday_window = _compute_holiday_window_dates(year=2024)
    train_df["holiday_flag"] = _compute_holiday_flag(train_df, holiday_window)
    test_df["holiday_flag"] = _compute_holiday_flag(test_df, holiday_window)

    # carrier_encoded: target encoding of OP_UNIQUE_CARRIER (mean delay rate per carrier, fit on train only)
    carrier_target_map = train_df.groupby("OP_UNIQUE_CARRIER", observed=False)[TARGET_COLUMN].mean().to_dict()
    train_df["carrier_encoded"] = train_df["OP_UNIQUE_CARRIER"].astype(str).map(carrier_target_map).fillna(global_delay_rate).astype("float32")
    test_df["carrier_encoded"] = test_df["OP_UNIQUE_CARRIER"].astype(str).map(carrier_target_map).fillna(global_delay_rate).astype("float32")

    # 10. Keep READABLE_COLUMNS in DataFrame alongside FEATURE_COLUMNS
    for col in READABLE_COLUMNS:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        else:
            train_df[col] = ""
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)
        else:
            test_df[col] = ""

    output_cols = [TARGET_COLUMN] + FEATURE_COLUMNS + [c for c in READABLE_COLUMNS if c not in FEATURE_COLUMNS]
    train_df = train_df[output_cols].copy()
    test_df = test_df[output_cols].copy()

    # 11. Save train_df to TRAIN_DATA_PATH as parquet
    os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    logger.info("Saved train DataFrame to %s", TRAIN_DATA_PATH)

    # 12. Save test_df to TEST_DATA_PATH as parquet
    os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    logger.info("Saved test DataFrame to %s", TEST_DATA_PATH)

    # 13. Save feature artifacts (scaler, lookup dicts) to outputs/feature_artifacts.joblib
    artifacts_path = "outputs/feature_artifacts.joblib"
    os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
    feature_artifacts = {
        "scaler": scaler,
        "route_frequency": route_frequency_map,
        "route_frequency_median": median_route_freq,
        "carrier_avg_delay": carrier_avg_delay_map,
        "origin_avg_delay": origin_avg_delay_map,
        "dest_avg_delay": dest_avg_delay_map,
        "carrier_encoded": carrier_target_map,
        "global_mean_delay": global_mean_delay,
        "global_delay_rate": global_delay_rate,
        "holiday_window": holiday_window,
        "feature_columns": FEATURE_COLUMNS,
        "readable_columns": READABLE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(feature_artifacts, artifacts_path)
    logger.info("Saved feature artifacts to %s", artifacts_path)

    # 14. Log: total rows, sample size, train rows, test rows, delay rates
    train_delay_rate = float(train_df[TARGET_COLUMN].mean()) if len(train_df) > 0 else 0.0
    test_delay_rate = float(test_df[TARGET_COLUMN].mean()) if len(test_df) > 0 else 0.0

    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info("Total rows loaded:       %d", total_raw_rows)
    logger.info("Non-cancelled rows:      %d", total_non_cancelled)
    logger.info("Sample size:             %d", len(df))
    logger.info("Train rows:              %d", len(train_df))
    logger.info("Test rows:               %d", len(test_df))
    logger.info("Train delay rate:        %.4f (%.2f%%)", train_delay_rate, train_delay_rate * 100)
    logger.info("Test delay rate:         %.4f (%.2f%%)", test_delay_rate, test_delay_rate * 100)
    logger.info("=" * 60)

    return train_df, test_df


if __name__ == "__main__":
    load_and_prepare("data/raw")
