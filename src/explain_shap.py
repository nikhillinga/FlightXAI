"""SHAP explainability module for FlightXAI.

Provides exact TreeExplainer SHAP interpretations for the trained XGBoost model,
generates global summary and dependence visualizations, and produces local
feature-level explanations in both structured and human-readable formats.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_EXPLAIN_SAMPLES,
    TRAIN_DATA_PATH,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_shap_explainer(model, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """Build exact TreeExplainer for XGBoost using background samples.

    Args:
        model: Trained XGBoost classifier.
        X_train: Training feature DataFrame.

    Returns:
        shap.TreeExplainer: Initialized SHAP tree explainer.
    """
    sample_size = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
    logger.info("Sampling %d background rows for TreeExplainer...", sample_size)
    background = shap.sample(X_train, sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model, data=background)
    logger.info("SHAP TreeExplainer successfully initialized.")
    return explainer


def compute_global_shap(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
) -> shap.Explanation:
    """Compute global SHAP values and save summary and dependence plots.

    Args:
        explainer: Trained TreeExplainer.
        X_sample: Evaluation dataset to compute SHAP values for.

    Returns:
        shap.Explanation: Global SHAP Explanation object.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sample_size = min(SHAP_EXPLAIN_SAMPLES, len(X_sample))
    if len(X_sample) > sample_size:
        X_eval = X_sample.sample(sample_size, random_state=RANDOM_STATE).copy()
    else:
        X_eval = X_sample.copy()

    logger.info("Computing SHAP values for %d sample rows...", len(X_eval))
    explanation = explainer(X_eval)

    # a) outputs/plots/shap_summary_bar.png — bar chart of mean |SHAP|
    bar_path = os.path.join(PLOTS_DIR, "shap_summary_bar.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation, show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary bar plot to %s", bar_path)

    # b) outputs/plots/shap_summary_dot.png — beeswarm plot
    dot_path = os.path.join(PLOTS_DIR, "shap_summary_dot.png")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(explanation, show=False)
    plt.title("SHAP Beeswarm Summary Plot", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(dot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary beeswarm plot to %s", dot_path)

    # c) outputs/plots/shap_dependence_top3.png — dependence plots for top 3 features
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    top3_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top3_features = [explanation.feature_names[i] for i in top3_indices]
    logger.info("Top 3 features by mean |SHAP|: %s", top3_features)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for feat, ax in zip(top3_features, axes):
        shap.plots.scatter(explanation[:, feat], ax=ax, show=False)
        ax.set_title(f"Dependence: {feat}", fontsize=12, pad=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    dep_path = os.path.join(PLOTS_DIR, "shap_dependence_top3.png")
    plt.suptitle("SHAP Dependence Plots for Top 3 Features", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved dependence plots to %s", dep_path)

    return explanation


def explain_single_shap(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
) -> dict[str, float]:
    """Explain a single flight record using SHAP.

    Args:
        explainer: Initialized TreeExplainer.
        X_row: Single-row pandas DataFrame containing FEATURE_COLUMNS.

    Returns:
        dict[str, float]: Feature contributions sorted by absolute SHAP value descending.
    """
    explanation = explainer(X_row)
    values = explanation.values[0]
    feature_names = X_row.columns.tolist()

    shap_dict = {
        feat: float(val)
        for feat, val in zip(feature_names, values)
    }
    return dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))


def get_plain_english_shap(
    shap_dict: dict[str, float],
    base_value: float | None = None,
    prediction: float | None = None,
) -> str:
    """Translate raw SHAP contribution values into a human-readable plain English summary.

    Args:
        shap_dict: Dictionary mapping feature names to SHAP values.
        base_value: Optional baseline model score/value.
        prediction: Optional model prediction score.

    Returns:
        str: Human-readable narrative explanation.
    """
    feature_labels = {
        "departure_hour": "departure hour",
        "day_of_week": "day of the week",
        "month": "flight month / seasonality",
        "is_weekend": "weekend schedule",
        "distance": "flight distance",
        "route_frequency": "route flight volume",
        "carrier_avg_delay": "carrier historical delay",
        "origin_avg_delay": "origin airport delay history",
        "dest_avg_delay": "destination airport delay history",
        "departure_peak_hour_flag": "peak departure hour schedule",
        "holiday_flag": "proximity to holiday",
        "carrier_encoded": "carrier delay risk level",
    }

    positives = [(k, v) for k, v in shap_dict.items() if v > 0]
    negatives = [(k, v) for k, v in shap_dict.items() if v < 0]

    top_pos = sorted(positives, key=lambda x: x[1], reverse=True)[:3]
    top_neg = sorted(negatives, key=lambda x: x[1])[:3]

    lines = []
    if top_pos:
        pos_items = [
            f"{feature_labels.get(feat, feat.replace('_', ' '))} (+{val:.2f})"
            for feat, val in top_pos
        ]
        lines.append("The main factors increasing delay probability are:\n  " + ",\n  ".join(pos_items) + ".")

    if top_neg:
        neg_items = [
            f"{feature_labels.get(feat, feat.replace('_', ' '))} ({val:.2f})"
            for feat, val in top_neg
        ]
        lines.append("The main factors reducing it are:\n  " + ",\n  ".join(neg_items) + ".")

    if not lines:
        return "All evaluated feature contributions were neutral."

    return "\n".join(lines)


if __name__ == "__main__":
    xgb_model_path = os.path.join(OUTPUT_DIR, "xgb_model.joblib")
    logger.info("Loading model from %s...", xgb_model_path)
    xgb_model = joblib.load(xgb_model_path)

    logger.info("Loading training features from %s...", TRAIN_DATA_PATH)
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    X_train = train_df[FEATURE_COLUMNS]

    # Build explainer
    explainer = build_shap_explainer(xgb_model, X_train)

    # Compute global SHAP and save plots
    logger.info("Computing global SHAP explanations and generating plots...")
    explanation = compute_global_shap(explainer, X_train)

    # Explain one example flight
    example_row = X_train.iloc[[0]]
    shap_dict = explain_single_shap(explainer, example_row)

    print("\n" + "=" * 65)
    print("SINGLE FLIGHT SHAP EXPLANATION (Top Features)")
    print("=" * 65)
    for feat, val in list(shap_dict.items())[:6]:
        print(f"  {feat:<25}: {val:+.4f}")
    print("=" * 65)

    base_val = float(explanation.base_values[0]) if hasattr(explanation, "base_values") else 0.0
    pred_prob = float(xgb_model.predict_proba(example_row)[:, 1][0])
    plain_english = get_plain_english_shap(shap_dict, base_value=base_val, prediction=pred_prob)

    print("\nPLAIN ENGLISH EXPLANATION:")
    print("-" * 65)
    print(plain_english)
    print("-" * 65 + "\n")
