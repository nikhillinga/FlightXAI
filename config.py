import os

# ── Paths ──────────────────────────────────────────────────────────────
RAW_DATA_DIR         = "data/raw"
TRAIN_DATA_PATH      = "data/processed/train.parquet"
TEST_DATA_PATH       = "data/processed/test.parquet"
OUTPUT_DIR           = "outputs"
PLOTS_DIR            = "outputs/plots"
REPORTS_DIR          = "outputs/reports"

# ── Sampling ───────────────────────────────────────────────────────────
SAMPLE_SIZE          = 500_000
RANDOM_STATE         = 42

# ── Data Split ─────────────────────────────────────────────────────────
TRAIN_YEAR           = 2024
TRAIN_MONTHS         = list(range(1, 10))
TEST_MONTHS          = list(range(10, 13))

# ── Model ──────────────────────────────────────────────────────────────
TARGET_COLUMN        = "delay"
XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

PREDICTION_THRESHOLD  = 0.30

FEATURE_COLUMNS = [
    "departure_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "distance",
    "route_frequency",
    "carrier_avg_delay",
    "origin_avg_delay",
    "dest_avg_delay",
    "departure_peak_hour_flag",
    "holiday_flag",
    "carrier_encoded",
]

READABLE_COLUMNS = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "ORIGIN_CITY_NAME",
    "DEST_CITY_NAME",
]

# ── SHAP ───────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 1000
SHAP_EXPLAIN_SAMPLES    = 100

# ── LIME ───────────────────────────────────────────────────────────────
LIME_NUM_FEATURES    = 8
LIME_NUM_SAMPLES     = 5000

# ── DiCE ───────────────────────────────────────────────────────────────
DICE_NUM_CF          = 3
DICE_FEATURES_TO_VARY = [
    "departure_hour",
    "month",
    "distance",
    "route_frequency",
]

# ── Evaluation ─────────────────────────────────────────────────────────
STABILITY_N_RUNS     = 5
SPARSITY_THRESHOLD   = 0.05
