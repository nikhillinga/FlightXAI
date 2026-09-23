"""DiCE counterfactual explanation module for FlightXAI.

Generates actionable "what-if" counterfactual explanations for flight delay
predictions using DiCE (Diverse Counterfactual Explanations).  Given a delayed
flight, DiCE suggests minimal feature changes that would flip the prediction
to on-time, helping stakeholders understand which operational levers matter.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import dice_ml
import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DICE_FEATURES_TO_VARY,
    DICE_NUM_CF,
    FEATURE_COLUMNS,
    OUTPUT_DIR,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────


def build_dice_explainer(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dice_ml.Dice:
    """Build a DiCE explainer around the trained model.

    Args:
        model: Trained sklearn-compatible classifier (e.g. XGBClassifier).
        X_train: Training feature DataFrame with FEATURE_COLUMNS.
        y_train: Training target Series (binary 0/1).

    Returns:
        dice_ml.Dice: Initialized DiCE explainer ready to generate
            counterfactual instances.
    """
    # Combine features + target into the single DataFrame DiCE expects
    train_df = pd.concat(
        [X_train[FEATURE_COLUMNS].reset_index(drop=True),
         y_train.reset_index(drop=True)],
        axis=1,
    )

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=FEATURE_COLUMNS,
        outcome_name=TARGET_COLUMN,
    )

    dice_model = dice_ml.Model(model=model, backend="sklearn")

    explainer = dice_ml.Dice(dice_data, dice_model, method="random")
    logger.info(
        "DiCE explainer built  —  %d training rows, %d features, method=random.",
        len(train_df),
        len(FEATURE_COLUMNS),
    )
    return explainer


def generate_counterfactuals(
    explainer: dice_ml.Dice,
    X_row: pd.DataFrame,
    desired_class: int = 0,
) -> pd.DataFrame:
    """Generate counterfactual explanations for a single flight.

    For a flight predicted as *delayed*, this answers: "What minimal changes
    would make the model predict *on-time*?"

    Args:
        explainer: Initialized DiCE explainer from :func:`build_dice_explainer`.
        X_row: Single-row DataFrame containing FEATURE_COLUMNS.
        desired_class: Target class for the counterfactuals (0 = on-time).

    Returns:
        pd.DataFrame: Comparison table with columns
            ``[feature, original_value, cf1_value, cf2_value, cf3_value]``
            filtered to only rows where at least one CF value differs from
            the original.
    """
    query = X_row[FEATURE_COLUMNS].copy()

    cf_result = explainer.generate_counterfactuals(
        query_instances=query,
        total_CFs=DICE_NUM_CF,
        desired_class=desired_class,
        features_to_vary=DICE_FEATURES_TO_VARY,
    )

    # Extract generated CF instances as a DataFrame
    cf_df = cf_result.cf_examples_list[0].final_cfs_df[FEATURE_COLUMNS]

    # Build the comparison table
    original_values = query.iloc[0]
    rows: list[dict] = []

    for feat in FEATURE_COLUMNS:
        row_data: dict = {
            "feature": feat,
            "original_value": original_values[feat],
        }
        for idx in range(len(cf_df)):
            row_data[f"cf{idx + 1}_value"] = cf_df.iloc[idx][feat]
        rows.append(row_data)

    result_df = pd.DataFrame(rows)

    # Keep only features where at least one CF value differs from original
    cf_cols = [c for c in result_df.columns if c.startswith("cf")]
    differs = result_df[cf_cols].apply(
        lambda col: ~np.isclose(col, result_df["original_value"], atol=1e-6),
    )
    mask = differs.any(axis=1)
    return result_df.loc[mask].reset_index(drop=True)


def explain_batch_dice(
    explainer: dice_ml.Dice,
    X_delayed: pd.DataFrame,
    n: int = 10,
) -> list[dict]:
    """Generate counterfactuals for a batch of delayed flights.

    Args:
        explainer: Initialized DiCE explainer.
        X_delayed: Feature DataFrame that may contain both delayed and
            on-time flights.  Only rows where the model predicts *delay == 1*
            are considered.
        n: Maximum number of flights to explain.

    Returns:
        list[dict]: One dict per explained flight containing:
            - ``"index"``: Original row index.
            - ``"original"``: Dict of original feature values.
            - ``"counterfactuals"``: The comparison DataFrame from
              :func:`generate_counterfactuals`.
    """
    # Filter to delayed predictions using the underlying model
    model = explainer.model.model
    preds = model.predict(X_delayed[FEATURE_COLUMNS])
    delayed_idx = X_delayed.index[preds == 1]

    sample_idx = delayed_idx[:n]
    logger.info(
        "Generating DiCE counterfactuals for %d / %d delayed flights...",
        len(sample_idx),
        len(delayed_idx),
    )

    results: list[dict] = []
    for idx in sample_idx:
        row = X_delayed.loc[[idx], FEATURE_COLUMNS]
        try:
            cf_table = generate_counterfactuals(explainer, row, desired_class=0)
            results.append({
                "index": idx,
                "original": row.iloc[0].to_dict(),
                "counterfactuals": cf_table,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate CFs for index %s: %s", idx, exc)

    logger.info("Batch DiCE done — %d explanations produced.", len(results))
    return results


def format_counterfactual_table(
    original_row: pd.Series | pd.DataFrame,
    cf_df: pd.DataFrame,
) -> str:
    """Format a counterfactual comparison as a human-readable string.

    Args:
        original_row: Original feature values (Series or single-row DataFrame).
        cf_df: Comparison DataFrame produced by :func:`generate_counterfactuals`.

    Returns:
        str: Multi-line plain-English recommendation.

    Example output::

        To reduce delay probability:
          - Change departure hour from 18 to 10
          - Change month from 7 to 3
    """
    if isinstance(original_row, pd.DataFrame):
        original_row = original_row.iloc[0]

    if cf_df.empty:
        return "No actionable changes were found by DiCE."

    lines: list[str] = ["To reduce delay probability:"]

    for _, row in cf_df.iterrows():
        feat = row["feature"]
        orig = row["original_value"]

        # Collect unique CF values that differ from original
        cf_vals = []
        for col in [c for c in cf_df.columns if c.startswith("cf")]:
            val = row[col]
            if not np.isclose(val, orig, atol=1e-6):
                cf_vals.append(val)

        # Deduplicate while preserving order
        seen: set[float] = set()
        unique_vals: list[float] = []
        for v in cf_vals:
            rounded = round(v, 2)
            if rounded not in seen:
                seen.add(rounded)
                unique_vals.append(v)

        for v in unique_vals:
            # Format integers cleanly (hours, months, flags, etc.)
            orig_fmt = _fmt(orig)
            cf_fmt = _fmt(v)
            lines.append(f"  - Change {feat} from {orig_fmt} to {cf_fmt}")

    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────


def _fmt(value: float) -> str:
    """Format a numeric value: show as int if it is one, else 2-decimal."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


# ── Main ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Load model
    xgb_model_path = os.path.join(OUTPUT_DIR, "xgb_model.joblib")
    logger.info("Loading model from %s...", xgb_model_path)
    xgb_model = joblib.load(xgb_model_path)

    # Load training data
    logger.info("Loading training data from %s...", TRAIN_DATA_PATH)
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    # Build DiCE explainer
    explainer = build_dice_explainer(xgb_model, X_train, y_train)

    # Load test data and find delayed flights
    logger.info("Loading test data from %s...", TEST_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    X_test = test_df[FEATURE_COLUMNS]

    preds = xgb_model.predict(X_test)
    delayed_mask = preds == 1
    delayed_indices = X_test.index[delayed_mask]

    if len(delayed_indices) == 0:
        logger.warning("No delayed flights found in test set — nothing to explain.")
        sys.exit(0)

    n_examples = min(3, len(delayed_indices))
    logger.info(
        "Found %d delayed flights in test set; explaining %d...",
        len(delayed_indices),
        n_examples,
    )

    print("\n" + "=" * 70)
    print("DiCE COUNTERFACTUAL EXPLANATIONS")
    print("=" * 70)

    for i, idx in enumerate(delayed_indices[:n_examples], start=1):
        row = X_test.loc[[idx]]
        prob = xgb_model.predict_proba(row)[:, 1][0]
        cf_table = generate_counterfactuals(explainer, row, desired_class=0)

        print(f"\n{'-' * 70}")
        print(f"Flight #{i}  (index={idx}, P(delay)={prob:.2%})")
        print(f"{'-' * 70}")

        if cf_table.empty:
            print("  (no counterfactuals found)")
            continue

        # Print comparison table
        header = f"  {'Feature':<25} {'Original':>10}"
        for c in [col for col in cf_table.columns if col.startswith("cf")]:
            header += f" {c:>10}"
        print(header)
        print("  " + "-" * len(header.strip()))

        for _, r in cf_table.iterrows():
            line = f"  {r['feature']:<25} {_fmt(r['original_value']):>10}"
            for c in [col for col in cf_table.columns if col.startswith("cf")]:
                line += f" {_fmt(r[c]):>10}"
            print(line)

        # Print plain English recommendation
        print()
        print(format_counterfactual_table(row, cf_table))

    print("\n" + "=" * 70 + "\n")
