# ✈️ FlightXAI — Explainable Flight Delay Prediction

FlightXAI is an end-to-end Explainable AI (XAI) system that predicts
U.S. domestic flight delays using XGBoost and then opens the black box with
three complementary explanation techniques: **SHAP** (global and local
feature attribution), **LIME** (local surrogate approximations), and
**DiCE** (diverse counterfactual "what-if" scenarios). A Streamlit
dashboard ties everything together, letting users enter a flight's
details and immediately see the prediction, the reasoning behind it, and
actionable changes that could reduce delay risk.

This is an academic project built to demonstrate the XAI syllabus
(Modules 2, 4, 5, 6) in a realistic, data-driven setting. It is
**separate from AeroFlow**, which targets the MLOps lifecycle — FlightXAI
focuses exclusively on **model interpretability and explanation quality**.
The two projects share the same BTS dataset but address different course
objectives.

---

## Architecture

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                       FlightXAI Pipeline                             │
  └───────────────────────────────────────────────────────────────────────┘

  data/raw/                src/preprocess.py         src/train.py
  ┌───────────┐            ┌───────────────┐         ┌─────────────┐
  │ BTS CSVs  │──────────▶ │ Clean, sample │────────▶│ XGBoost     │
  │ 2024 Jan– │            │ engineer feat │         │ + Logistic  │
  │  Dec      │            │ train / test  │         │ Regression  │
  └───────────┘            └───────────────┘         └──────┬──────┘
                                  │                         │
                                  ▼                         ▼
                          outputs/feature_      outputs/xgb_model.joblib
                          artifacts.joblib      outputs/lr_model.joblib
                                                        │
                    ┌───────────────────────────────────┐│
                    │         Explanation Layer         ││
                    │  ┌────────┐ ┌──────┐ ┌────────┐  ││
                    │  │ SHAP   │ │ LIME │ │ DiCE   │  ││
                    │  │ Tree-  │ │ Tab- │ │ Random │◀─┘│
                    │  │ Expl.  │ │ Expl.│ │ CF Gen │   │
                    │  └───┬────┘ └──┬───┘ └───┬────┘   │
                    └──────┼────────-┼─────────┼────────┘
                           │        │         │
                           ▼        ▼         ▼
                    ┌──────────────────────────────────┐
                    │       Streamlit Dashboard         │
                    │  Tab 1: Prediction                │
                    │  Tab 2: Why? (SHAP + LIME)        │
                    │  Tab 3: What if? (DiCE)           │
                    └──────────────────────────────────┘
                          http://localhost:8501
```

---

## Tech Stack

| Component          | Tool / Library            |
| :----------------- | :------------------------ |
| Language           | Python 3.10+              |
| Prediction (main)  | XGBoost 2.0               |
| Prediction (baseline) | Logistic Regression (scikit-learn) |
| Global + local XAI | SHAP 0.45 (TreeExplainer) |
| Local surrogate    | LIME 0.2                  |
| Counterfactuals    | DiCE-ML 0.12              |
| Dashboard          | Streamlit 1.35 + Plotly 5 |
| Data format        | Parquet (via PyArrow)     |
| Preprocessing      | pandas, scikit-learn StandardScaler |
| Visualization      | Plotly, Matplotlib, Seaborn |
| Linting            | Ruff                      |
| Testing            | pytest                    |

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add data

Copy (or symlink) the **2024 BTS monthly CSVs** into `data/raw/`:

```
data/raw/
├── 2024_Jan.csv
├── 2024_Feb.csv
├── ...
└── 2024_Dec.csv
```

> These are the same files used in AeroFlow. Download from
> [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_id=07ebeefc-dafe-40b7-a1ca-f0c27ad30800&Table_ID=236&Has_Group=3&Is_G498=false)
> if you don't already have them.

### 3. Run the pipeline

```bash
python src/preprocess.py       # Clean → feature-engineer → train/test split
python src/train.py            # Train XGBoost + Logistic Regression
python src/explain_shap.py     # Global SHAP plots + single-flight example
python src/explain_lime.py     # LIME example explanation + plot
python src/explain_dice.py     # DiCE counterfactual examples
python src/evaluate.py         # Fidelity, Stability, Sparsity, CF Validity
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## Example Outputs

| Module                | What It Produces                                                                   |
| :-------------------- | :--------------------------------------------------------------------------------- |
| `preprocess.py`       | `data/processed/train.parquet`, `test.parquet`, `outputs/feature_artifacts.joblib` |
| `train.py`            | `outputs/xgb_model.joblib`, `lr_model.joblib`, `model_metrics.json`, `outputs/plots/roc_curve.png` |
| `explain_shap.py`     | `outputs/plots/shap_summary_bar.png`, `shap_summary_dot.png`, `shap_dependence_top3.png` |
| `explain_lime.py`     | `outputs/plots/lime_example.png`                                                  |
| `explain_dice.py`     | Console table of counterfactual scenarios for 3 delayed flights                   |
| `evaluate.py`         | `outputs/reports/evaluation_report.json` + console summary table                  |
| `dashboard/app.py`    | Interactive Streamlit app with Prediction / Why? / What if? tabs                  |

---

## Explanation Metrics

The evaluation suite (`src/evaluate.py`) quantifies explanation quality
across four dimensions:

| Metric                         | What It Measures                                                   | Target   |
| :----------------------------- | :----------------------------------------------------------------- | :------- |
| **Fidelity** (LIME vs XGBoost) | How closely LIME's local surrogate matches XGBoost's probability   | > 0.90   |
| **Stability** (SHAP)          | Consistency of SHAP values when the background sample changes      | > 0.95   |
| **Sparsity** (SHAP)           | Conciseness — fraction of features with negligible SHAP values     | > 0.50   |
| **CF Validity** (DiCE)        | Proportion of counterfactuals that actually flip the prediction    | > 0.80   |

---

## XAI Syllabus Coverage

| Module | Topic                              | Project Component                                                    |
| :----- | :--------------------------------- | :------------------------------------------------------------------- |
| 2      | Interpretable Models & Feature Attribution | SHAP TreeExplainer (global bar/beeswarm + local waterfall), LIME local surrogate |
| 4      | Local Explanations                 | `explain_single_shap()`, `explain_single_lime()`, plain-English summaries, Dashboard "Why?" tab |
| 5      | Counterfactual Explanations        | DiCE random CF generation, `generate_counterfactuals()`, Dashboard "What if?" tab |
| 6      | Evaluating Explanations            | `evaluate.py` — Fidelity, Stability, Sparsity, CF Validity with pass/fail targets |

---

## Key Design Decisions

- **XGBoost as the primary model.** Non-linear tree ensemble captures
  complex delay patterns that Logistic Regression misses; SHAP's
  TreeExplainer provides exact (not approximate) Shapley values for trees.

- **Threshold = 0.30 instead of 0.50.** Flight delays are costly to miss;
  a lower threshold improves recall at the expense of precision, which
  matches the operational preference for early warnings.

- **StandardScaler on distance only.** Other features are counts, flags,
  or ordinals that don't need scaling — applying it only to distance
  keeps explanations more interpretable.

- **Target encoding for carrier.** `carrier_encoded` uses the per-carrier
  delay rate from training data, avoiding one-hot explosion while
  preserving ordinal delay-risk information for SHAP.

- **DiCE varies only actionable features.** `DICE_FEATURES_TO_VARY` is
  restricted to `departure_hour`, `month`, `distance`, and
  `route_frequency` — factors a traveller could realistically change.

- **Three explainers, one dashboard.** SHAP and LIME provide independent
  corroboration of feature importance; DiCE adds actionable
  recommendations. Showing all three together lets users triangulate the
  explanation.

---

## Project Structure

```
FlightXAI/
├── config.py                  # All hyperparameters, paths, feature lists
├── requirements.txt
├── README.md
├── decisions.md               # Design decision log
│
├── data/
│   ├── raw/                   # BTS CSV files (gitignored)
│   └── processed/             # train.parquet, test.parquet (gitignored)
│
├── src/
│   ├── preprocess.py          # Data loading, cleaning, feature engineering
│   ├── train.py               # XGBoost + Logistic Regression training
│   ├── explain_shap.py        # SHAP global + local explanations
│   ├── explain_lime.py        # LIME local explanations
│   ├── explain_dice.py        # DiCE counterfactual generation
│   └── evaluate.py            # Explanation quality metrics
│
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
│
├── outputs/                   # Models, plots, reports (gitignored)
│   ├── xgb_model.joblib
│   ├── lr_model.joblib
│   ├── feature_artifacts.joblib
│   ├── model_metrics.json
│   ├── plots/
│   └── reports/
│
├── notebooks/                 # Exploratory analysis
└── tests/                     # Unit tests
```

---

## Note on Data

The raw BTS data files are **not included** in this repository (they are
gitignored at `data/raw/` and `data/processed/`). To reproduce:

1. Download the "Reporting Carrier On-Time Performance" dataset for all
   months of **2024** from
   [transtats.bts.gov](https://www.transtats.bts.gov/).
2. Place the monthly CSVs in `data/raw/`.
3. Run `python src/preprocess.py` to generate the processed parquet files.

If you already have the data from the **AeroFlow** project, you can
symlink or copy the same CSVs — both projects use identical source files.
