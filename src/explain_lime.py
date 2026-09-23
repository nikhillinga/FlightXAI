"""LIME explainability module for FlightXAI.

Provides local tabular explanations for flight delay predictions using LIME
(Local Interpretable Model-agnostic Explanations), supporting single-flight
interpretations, batch stability evaluations, and visual figure generation.
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
from lime.lime_tabular import LimeTabularExplainer

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TRAIN_DATA_PATH,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_lime_explainer(X_train: pd.DataFrame | np.ndarray) -> LimeTabularExplainer:
    """Build a LimeTabularExplainer for classification on flight features.

    Args:
        X_train: Training feature DataFrame or numpy array.

    Returns:
        LimeTabularExplainer: Fitted tabular LIME explainer.
    """
    train_values = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)

    logger.info("Initializing LimeTabularExplainer with %d background rows...", len(train_values))
    explainer = LimeTabularExplainer(
        training_data=train_values,
        feature_names=FEATURE_COLUMNS,
        class_names=["on_time", "delayed"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    logger.info("LimeTabularExplainer initialized successfully.")
    return explainer


def explain_single_lime(
    explainer: LimeTabularExplainer,
    model,
    X_row: pd.DataFrame | pd.Series | np.ndarray,
) -> dict[str, float]:
    """Explain a single flight record using LIME.

    Args:
        explainer: Initialized LimeTabularExplainer.
        model: Trained model with predict_proba method.
        X_row: Single flight record features (1-row DataFrame, Series, or 1D array).

    Returns:
        dict[str, float]: Top LIME_NUM_FEATURES feature weights sorted by |weight| descending.
    """
    if isinstance(X_row, pd.DataFrame):
        row_values = X_row.iloc[0].values
    elif isinstance(X_row, pd.Series):
        row_values = X_row.values
    else:
        row_values = np.asarray(X_row).ravel()

    exp = explainer.explain_instance(
        data_row=row_values,
        predict_fn=model.predict_proba,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        labels=(1,),
    )

    feature_names = explainer.feature_names
    weights_dict = {
        feature_names[feat_idx]: float(weight)
        for feat_idx, weight in exp.as_map()[1]
    }

    return dict(sorted(weights_dict.items(), key=lambda item: abs(item[1]), reverse=True))


def explain_batch_lime(
    explainer: LimeTabularExplainer,
    model,
    X_sample: pd.DataFrame | np.ndarray,
    n: int = 20,
) -> list[dict[str, float]]:
    """Explain a batch of n flight records using LIME.

    Args:
        explainer: Initialized LimeTabularExplainer.
        model: Trained classifier with predict_proba method.
        X_sample: Evaluation dataset.
        n: Number of records to explain (default: 20).

    Returns:
        list[dict[str, float]]: List of explanation dictionaries (one per row).
    """
    sample_count = min(n, len(X_sample))
    if isinstance(X_sample, pd.DataFrame):
        subset = X_sample.iloc[:sample_count]
    else:
        subset = X_sample[:sample_count]

    logger.info("Generating batch LIME explanations for %d records...", sample_count)
    batch_explanations: list[dict[str, float]] = []

    for i in range(sample_count):
        row = subset.iloc[[i]] if isinstance(subset, pd.DataFrame) else subset[i]
        weights = explain_single_lime(explainer, model, row)
        batch_explanations.append(weights)

    return batch_explanations


def save_lime_plot(
    explainer: LimeTabularExplainer,
    model,
    X_row: pd.DataFrame | pd.Series | np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "lime_example.png"),
) -> None:
    """Generate and save LIME explanation plot for one flight record.

    Args:
        explainer: Initialized LimeTabularExplainer.
        model: Trained classifier with predict_proba.
        X_row: Single flight record.
        output_path: Path to save the output PNG file.
    """
    if isinstance(X_row, pd.DataFrame):
        row_values = X_row.iloc[0].values
    elif isinstance(X_row, pd.Series):
        row_values = X_row.values
    else:
        row_values = np.asarray(X_row).ravel()

    exp = explainer.explain_instance(
        data_row=row_values,
        predict_fn=model.predict_proba,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        labels=(1,),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = exp.as_pyplot_figure(label=1)
    plt.title("LIME Local Explanation — Flight Delay Attribution", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LIME explanation plot to %s", output_path)


if __name__ == "__main__":
    xgb_model_path = os.path.join(OUTPUT_DIR, "xgb_model.joblib")
    logger.info("Loading model from %s...", xgb_model_path)
    xgb_model = joblib.load(xgb_model_path)

    logger.info("Loading training features from %s...", TRAIN_DATA_PATH)
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    X_train = train_df[FEATURE_COLUMNS]

    # Build LIME explainer
    explainer = build_lime_explainer(X_train)

    # Explain one example flight
    example_row = X_train.iloc[[0]]
    lime_weights = explain_single_lime(explainer, xgb_model, example_row)

    print("\n" + "=" * 65)
    print("SINGLE FLIGHT LIME EXPLANATION (Top Feature Weights)")
    print("=" * 65)
    for feat, weight in lime_weights.items():
        print(f"  {feat:<25}: {weight:+.4f}")
    print("=" * 65 + "\n")

    # Save LIME plot
    lime_plot_path = os.path.join(PLOTS_DIR, "lime_example.png")
    save_lime_plot(explainer, xgb_model, example_row, lime_plot_path)
