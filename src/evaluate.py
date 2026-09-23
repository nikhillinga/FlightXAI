"""Explainability evaluation module for FlightXAI.

Quantifies the quality of SHAP, LIME, and DiCE explanations using four
complementary metrics: fidelity, stability, sparsity, and counterfactual
validity.  Results are persisted as a JSON report and printed to the console.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DICE_NUM_CF,
    FEATURE_COLUMNS,
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    OUTPUT_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_EXPLAIN_SAMPLES,
    SPARSITY_THRESHOLD,
    STABILITY_N_RUNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.explain_dice import build_dice_explainer, generate_counterfactuals
from src.explain_lime import build_lime_explainer
from src.explain_shap import build_shap_explainer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Individual Metric Functions ────────────────────────────────────────


def _compute_fidelity(
    model,
    lime_explainer: LimeTabularExplainer,
    X_test: pd.DataFrame,
    n: int = 100,
) -> dict:
    """Fidelity: how closely LIME's local model matches XGBoost.

    For each sampled row the LIME surrogate intercept + weights gives a
    local predicted probability.  We compare that against XGBoost's
    ``predict_proba`` and report ``1 - MAE``.

    Returns:
        dict with ``score`` and ``details``.
    """
    sample_size = min(n, len(X_test))
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(X_test), size=sample_size, replace=False)

    abs_errors: list[float] = []

    for idx in indices:
        row_values = X_test.iloc[idx].values

        # LIME local explanation — label 1 (delayed)
        exp = lime_explainer.explain_instance(
            data_row=row_values,
            predict_fn=model.predict_proba,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
            labels=(1,),
        )

        # LIME local linear model prediction for class 1
        lime_prob = float(exp.local_pred[0])

        # XGBoost probability for class 1
        xgb_prob = float(model.predict_proba(row_values.reshape(1, -1))[:, 1][0])

        abs_errors.append(abs(lime_prob - xgb_prob))

    mae = float(np.mean(abs_errors))
    score = round(1.0 - mae, 4)

    return {
        "score": score,
        "details": {
            "n_samples": sample_size,
            "mean_absolute_error": round(mae, 6),
            "std_absolute_error": round(float(np.std(abs_errors)), 6),
            "target": "> 0.90",
        },
    }


def _compute_stability(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_rows: int = 20,
    n_runs: int = STABILITY_N_RUNS,
) -> dict:
    """Stability: consistency of SHAP values across repeated runs.

    Rebuilds the TreeExplainer with different background samples each run
    and measures how much SHAP values fluctuate.

    Returns:
        dict with ``score`` and ``details``.
    """
    sample_size = min(n_rows, len(X_test))
    rng = np.random.RandomState(RANDOM_STATE)
    row_indices = rng.choice(len(X_test), size=sample_size, replace=False)
    X_eval = X_test.iloc[row_indices].copy()

    n_features = len(FEATURE_COLUMNS)

    # Shape: (n_runs, n_rows, n_features)
    all_shap_values = np.zeros((n_runs, sample_size, n_features))

    for run in range(n_runs):
        seed = RANDOM_STATE + run + 1
        bg_size = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
        background = shap.sample(X_train, bg_size, random_state=seed)
        explainer = shap.TreeExplainer(model, data=background)

        explanation = explainer(X_eval)
        all_shap_values[run] = explanation.values

        logger.info("  Stability run %d/%d done (seed=%d).", run + 1, n_runs, seed)

    # Std across runs for each (row, feature), then mean over all
    std_across_runs = np.std(all_shap_values, axis=0)  # (n_rows, n_features)
    mean_std = float(np.mean(std_across_runs))
    score = round(1.0 - mean_std, 4)

    # Per-feature mean std
    per_feature_std = {
        feat: round(float(std_across_runs[:, i].mean()), 6)
        for i, feat in enumerate(FEATURE_COLUMNS)
    }

    return {
        "score": score,
        "details": {
            "n_rows": sample_size,
            "n_runs": n_runs,
            "mean_std_across_runs": round(mean_std, 6),
            "per_feature_std": per_feature_std,
            "target": "> 0.95",
        },
    }


def _compute_sparsity(
    shap_explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    n: int = SHAP_EXPLAIN_SAMPLES,
) -> dict:
    """Sparsity: conciseness of SHAP explanations.

    For each row, counts features whose |SHAP value| exceeds
    ``SPARSITY_THRESHOLD``.  Higher sparsity means fewer dominant features
    (more concise explanations).

    Returns:
        dict with ``score`` and ``details``.
    """
    sample_size = min(n, len(X_test))
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(X_test), size=sample_size, replace=False)
    X_eval = X_test.iloc[indices].copy()

    explanation = shap_explainer(X_eval)
    shap_values = explanation.values  # (n_samples, n_features)

    n_features = len(FEATURE_COLUMNS)
    active_counts = np.sum(np.abs(shap_values) > SPARSITY_THRESHOLD, axis=1)
    mean_active = float(np.mean(active_counts))
    score = round(1.0 - (mean_active / n_features), 4)

    return {
        "score": score,
        "details": {
            "n_samples": sample_size,
            "total_features": n_features,
            "mean_active_features": round(mean_active, 2),
            "sparsity_threshold": SPARSITY_THRESHOLD,
            "target": "> 0.50",
        },
    }


def _compute_counterfactual_validity(
    model,
    dice_explainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n: int = 20,
) -> dict:
    """Counterfactual validity: proportion of CFs that flip the prediction.

    Generates counterfactuals for delayed flights (true label == 1) and
    checks whether XGBoost actually predicts class 0 for each one.

    Returns:
        dict with ``score`` and ``details``.
    """
    delayed_mask = y_test == 1
    X_delayed = X_test.loc[delayed_mask]

    sample_size = min(n, len(X_delayed))
    rng = np.random.RandomState(RANDOM_STATE)
    sample_indices = rng.choice(len(X_delayed), size=sample_size, replace=False)
    X_sample = X_delayed.iloc[sample_indices]

    total_cfs = 0
    valid_cfs = 0
    per_flight: list[dict] = []

    for idx in X_sample.index:
        row = X_sample.loc[[idx], FEATURE_COLUMNS]
        try:
            cf_table = generate_counterfactuals(dice_explainer, row, desired_class=0)

            if cf_table.empty:
                per_flight.append({"index": int(idx), "n_cfs": 0, "n_valid": 0})
                continue

            # Reconstruct full CF feature vectors
            cf_cols = [c for c in cf_table.columns if c.startswith("cf")]
            n_cf = len(cf_cols)

            for cf_col in cf_cols:
                # Start from original and overlay changed features
                cf_vector = row.iloc[0].copy()
                for _, r in cf_table.iterrows():
                    cf_vector[r["feature"]] = r[cf_col]

                pred = model.predict(cf_vector.values.reshape(1, -1))[0]
                total_cfs += 1
                if pred == 0:
                    valid_cfs += 1

            per_flight.append({"index": int(idx), "n_cfs": n_cf, "n_valid": valid_cfs})

        except Exception as exc:  # noqa: BLE001
            logger.warning("CF generation failed for index %s: %s", idx, exc)
            per_flight.append({"index": int(idx), "n_cfs": 0, "n_valid": 0, "error": str(exc)})

    score = round(valid_cfs / total_cfs, 4) if total_cfs > 0 else 0.0

    return {
        "score": score,
        "details": {
            "n_flights": sample_size,
            "total_counterfactuals": total_cfs,
            "valid_counterfactuals": valid_cfs,
            "dice_num_cf": DICE_NUM_CF,
            "target": "> 0.80",
        },
    }


# ── Main Public Function ──────────────────────────────────────────────


def run_evaluation(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_explainer: shap.TreeExplainer,
    lime_explainer: LimeTabularExplainer,
    dice_explainer,
) -> dict:
    """Run the full explainability evaluation suite.

    Computes fidelity, stability, sparsity, and counterfactual validity
    metrics, persists a JSON report, and prints a summary table.

    Args:
        model: Trained XGBoost classifier.
        X_train: Training features (used for SHAP background re-sampling).
        X_test: Test features.
        y_test: Test labels (binary 0/1).
        shap_explainer: Initialized SHAP TreeExplainer.
        lime_explainer: Initialized LIME TabularExplainer.
        dice_explainer: Initialized DiCE explainer.

    Returns:
        dict: Evaluation results with top-level scores, timestamp, and details.
    """
    # 1. Fidelity (LIME vs XGBoost)
    logger.info("=" * 60)
    logger.info("[1/4] Computing FIDELITY (LIME vs XGBoost)...")
    fidelity = _compute_fidelity(model, lime_explainer, X_test, n=100)
    logger.info("  Fidelity score: %.4f", fidelity["score"])

    # 2. Stability (SHAP)
    logger.info("=" * 60)
    logger.info("[2/4] Computing STABILITY (SHAP across %d runs)...", STABILITY_N_RUNS)
    stability = _compute_stability(model, X_train, X_test, n_rows=20, n_runs=STABILITY_N_RUNS)
    logger.info("  Stability score: %.4f", stability["score"])

    # 3. Sparsity (SHAP)
    logger.info("=" * 60)
    logger.info("[3/4] Computing SPARSITY (SHAP)...")
    sparsity = _compute_sparsity(shap_explainer, X_test, n=SHAP_EXPLAIN_SAMPLES)
    logger.info("  Sparsity score: %.4f", sparsity["score"])

    # 4. Counterfactual validity (DiCE)
    logger.info("=" * 60)
    logger.info("[4/4] Computing COUNTERFACTUAL VALIDITY (DiCE)...")
    cf_validity = _compute_counterfactual_validity(model, dice_explainer, X_test, y_test, n=20)
    logger.info("  CF Validity score: %.4f", cf_validity["score"])

    # Assemble report
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "fidelity": fidelity["score"],
        "stability": stability["score"],
        "sparsity": sparsity["score"],
        "counterfactual_validity": cf_validity["score"],
        "timestamp": timestamp,
        "details": {
            "fidelity": fidelity["details"],
            "stability": stability["details"],
            "sparsity": sparsity["details"],
            "counterfactual_validity": cf_validity["details"],
        },
    }

    # Persist report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved evaluation report to %s", report_path)

    # Print summary table
    _print_summary(report)

    return report


# ── Helpers ────────────────────────────────────────────────────────────


def _print_summary(report: dict) -> None:
    """Print a clean evaluation summary table to stdout."""
    metrics = [
        ("Fidelity (LIME vs XGBoost)", report["fidelity"], "> 0.90"),
        ("Stability (SHAP)",           report["stability"], "> 0.95"),
        ("Sparsity (SHAP)",            report["sparsity"], "> 0.50"),
        ("CF Validity (DiCE)",         report["counterfactual_validity"], "> 0.80"),
    ]

    col_metric = 34
    col_score = 10
    col_target = 10
    col_status = 8
    header_width = col_metric + col_score + col_target + col_status + 6

    targets_num = [0.90, 0.95, 0.50, 0.80]

    print("\n" + "=" * header_width)
    print("  EXPLAINABILITY EVALUATION REPORT")
    print("=" * header_width)
    print(
        f"  {'Metric':<{col_metric}}"
        f"  {'Score':>{col_score}}"
        f"  {'Target':>{col_target}}"
        f"  {'Status':>{col_status}}"
    )
    print("  " + "-" * (header_width - 2))

    for (name, score, target), threshold in zip(metrics, targets_num):
        status = "PASS" if score >= threshold else "FAIL"
        print(
            f"  {name:<{col_metric}}"
            f"  {score:>{col_score}.4f}"
            f"  {target:>{col_target}}"
            f"  {status:>{col_status}}"
        )

    print("=" * header_width)
    print(f"  Timestamp: {report['timestamp']}")
    print("=" * header_width + "\n")


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

    # Load test data
    logger.info("Loading test data from %s...", TEST_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Build all three explainers
    logger.info("Building SHAP explainer...")
    shap_exp = build_shap_explainer(xgb_model, X_train)

    logger.info("Building LIME explainer...")
    lime_exp = build_lime_explainer(X_train)

    logger.info("Building DiCE explainer...")
    dice_exp = build_dice_explainer(xgb_model, X_train, y_train)

    # Run full evaluation
    logger.info("Starting full evaluation suite...")
    results = run_evaluation(
        model=xgb_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        shap_explainer=shap_exp,
        lime_explainer=lime_exp,
        dice_explainer=dice_exp,
    )
