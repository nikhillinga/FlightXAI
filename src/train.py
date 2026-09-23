"""Model training module for FlightXAI.

Trains Logistic Regression and XGBoost classifiers on BTS flight delay features,
evaluates performance using standard classification metrics at threshold 0.30,
generates a ROC comparison curve, and persists trained models and metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    XGB_PARAMS,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PREDICTION_THRESHOLD = 0.30


def _evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = PREDICTION_THRESHOLD,
) -> tuple[dict[str, float], pd.Series]:
    """Compute classification metrics and predicted probabilities for a model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
    }
    return metrics, y_prob


def train_models() -> dict:
    """Train Logistic Regression and XGBoost classifiers, evaluate and persist outputs.

    Returns:
        dict: Metrics comparison dictionary for both models.
    """
    # 1. Load train and test parquets
    logger.info("Loading training data from %s...", TRAIN_DATA_PATH)
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    logger.info("Loading test data from %s...", TEST_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    # 2. Separate X and y using FEATURE_COLUMNS and TARGET_COLUMN
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info(
        "Data split loaded: X_train shape=%s, X_test shape=%s",
        X_train.shape,
        X_test.shape,
    )

    # 3. Train Model 1 — Logistic Regression
    logger.info("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_metrics, lr_prob = _evaluate_model(lr_model, X_test, y_test)
    logger.info("Logistic Regression evaluation complete: %s", lr_metrics)

    # 4. Train Model 2 — XGBoost
    logger.info("Training XGBoost model...")
    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    xgb_metrics, xgb_prob = _evaluate_model(xgb_model, X_test, y_test)
    logger.info("XGBoost evaluation complete: %s", xgb_metrics)

    # 5. Compile metrics dictionary
    metrics = {
        "Logistic Regression": lr_metrics,
        "XGBoost": xgb_metrics,
    }

    # 6. Print clean metrics comparison table
    comparison_df = pd.DataFrame(metrics).T
    print("\n" + "=" * 70)
    print(f"{'MODEL PERFORMANCE COMPARISON (Threshold = ' + str(PREDICTION_THRESHOLD) + ')':^70}")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70 + "\n")

    # 7. Save models
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lr_path = os.path.join(OUTPUT_DIR, "lr_model.joblib")
    xgb_path = os.path.join(OUTPUT_DIR, "xgb_model.joblib")

    joblib.dump(lr_model, lr_path)
    logger.info("Saved Logistic Regression model to %s", lr_path)

    joblib.dump(xgb_model, xgb_path)
    logger.info("Saved XGBoost model to %s", xgb_path)

    # 8. Save metrics to outputs/model_metrics.json
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # 9. Save ROC curve plot comparing both models
    os.makedirs(PLOTS_DIR, exist_ok=True)
    roc_plot_path = os.path.join(PLOTS_DIR, "roc_curve.png")

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_lr,
        tpr_lr,
        label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})",
        color="#1f77b4",
        lw=2,
    )
    plt.plot(
        fpr_xgb,
        tpr_xgb,
        label=f"XGBoost (AUC = {xgb_metrics['roc_auc']:.3f})",
        color="#ff7f0e",
        lw=2,
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve Comparison — Flight Delay Prediction", fontsize=13, pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve comparison plot to %s", roc_plot_path)

    return metrics


if __name__ == "__main__":
    train_models()
