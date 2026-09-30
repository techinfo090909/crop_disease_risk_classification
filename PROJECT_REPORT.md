# Crop Disease Risk Classification
### A Machine Learning Approach to Agricultural Risk Assessment

**Capstone Project — Learn Depth Academy LLP, Track 1, Problem 16**
**Domain:** Agriculture &nbsp;|&nbsp; **Task:** Multi-class Classification &nbsp;|&nbsp; **Duration:** 10 Days

---

## Abstract

Crop diseases cause significant yield loss worldwide, and early risk assessment allows growers to intervene before an outbreak spreads. This project develops a zero-cost, end-to-end machine learning pipeline that classifies crop disease risk into three categories — **Low**, **Medium**, and **High** — using crop type, target disease, growth stage, and environmental readings such as temperature, humidity, rainfall, leaf wetness, wind speed, and soil moisture. Three foundational classifiers (Logistic Regression, Decision Tree, and K-Nearest Neighbors) were trained on a balanced 10,000-row dataset and compared using accuracy, precision, recall, and macro F1-score. The K-Nearest Neighbors model was selected as the best performer (macro F1 = 0.681) and deployed through an interactive Streamlit application that returns a risk category along with a full class-probability breakdown.

---

## 1. Introduction

Timely detection of disease-favourable conditions is one of the most cost-effective forms of crop protection: a grower who knows conditions are trending toward "High risk" can act — inspecting fields, adjusting irrigation, or applying preventive treatment — before symptoms are visible. This project frames that problem as a supervised, multi-class classification task and builds a complete, reproducible pipeline around it: data validation, exploratory data analysis, model training and comparison, and a lightweight prediction interface, using only free and open-source tools within a 10-day scope.

---

## 2. Problem Definition

**Given** a crop, a target disease, the crop's current growth stage, and a set of environmental readings for a given day, **predict** the disease-risk category (Low / Medium / High) for that observation, along with the model's confidence in each category.

This is treated strictly as an educational prototype: the goal is a defensible, well-evaluated foundational model and a working demonstration, not a production-grade agronomic diagnostic system.

---

## 3. Related Work

Environmental disease-forecasting models in agriculture typically rely on **weather-based infection models** (e.g., temperature–leaf-wetness duration tables used for late blight and other fungal diseases), where a pathogen requires a minimum combination of temperature and continuous leaf wetness to infect a host. Rather than hand-encoding such domain rules, this project takes a **data-driven approach**: foundational supervised classifiers are trained directly on environmental and crop features to learn the risk relationship empirically, which is faster to build within the project's constraints and generalizes across the five crop–disease pairs in scope.

---

## 4. Dataset

| Property | Value |
|---|---|
| Rows | 10,000 |
| Raw features | 11 (`crop`, `disease`, `day_of_year`, `avg_temp_c`, `avg_humidity_pct`, `rainfall_mm`, `leaf_wetness_hours`, `wind_speed_kmph`, `soil_moisture_pct`, `prior_regional_incidence_pct`, `growth_stage`) |
| Target | `risk_label` (Low / Medium / High), derived from a numeric `risk_score` |
| Missing values | 0 |
| Duplicate rows | 0 |
| Crops | Cotton, Rice, Potato, Wheat, Grape — 2,000 rows each |
| Diseases | Bacterial Blight, Blast, Late Blight, Stripe Rust, Powdery Mildew — 2,000 rows each |

**`growth_stage`** is engineered, not independently collected: it is a deterministic bucketing of `day_of_year` (Seedling ≤ day 90, Vegetative ≤ day 180, Flowering ≤ day 270, Maturity > day 270). It is included as a categorical view of seasonal timing that some models can exploit differently than the raw numeric day, but it does not add information beyond `day_of_year`.

### 4.1 Target distribution

| Category | Count | Percentage |
|---|---|---|
| Medium | 4,669 | 46.69% |
| Low | 3,574 | 35.74% |
| High | 1,757 | 17.57% |

The target is moderately imbalanced, with High-risk observations under-represented relative to Low and Medium. This was accounted for by using **macro-averaged** precision, recall, and F1-score (which weight all three classes equally) rather than accuracy alone when comparing models.

### 4.2 Source and provenance — ⚠️ citation still to be confirmed

Direct inspection of the delivered CSV **confirms this is a synthetic, formula-generated dataset**, not raw field-collected data:

- Every crop and every disease category appears with an **identical count — exactly 2,000 of the 10,000 rows each**.
- **Zero missing values and zero duplicate rows** across the entire file.
- `risk_label` is not an observed outcome — it is a **deterministic threshold cut** of `risk_score`:
  - Low: `risk_score` in [0, 0.35)
  - Medium: `risk_score` in [0.35, 0.65)
  - High: `risk_score` in [0.65, 1.0]
- Correlation of `risk_score` with the raw features shows `prior_regional_incidence_pct` (r = 0.73) and `avg_humidity_pct` (r = 0.26) as by far the strongest drivers, with the remaining six numerical features contributing weakly (|r| < 0.1) — consistent with a weighted-formula construction rather than an empirically measured index.

Per capstone rules, a derived/synthetic target must be explicitly disclosed (done above) and the dataset must be credited to its actual origin. **What remains is the origin itself** — the exact source should be added here before this report is finalized:

> **Dataset source:** *[blob:https://github.com/39bb750c-f292-470c-a58b-f84e412dc272]*

---

## 5. Methodology

1. **Data validation** — schema check, missing-value scan, duplicate scan, class-balance check (`data_validation.py`).
2. **Feature engineering** — `growth_stage` derived from `day_of_year`.
3. **Preprocessing pipeline** (`scikit-learn` `ColumnTransformer`, fit only on the training split):
   - Categorical features (`crop`, `disease`, `growth_stage`) → `OneHotEncoder`
   - Numerical features (`day_of_year`, `avg_temp_c`, `avg_humidity_pct`, `rainfall_mm`, `leaf_wetness_hours`, `wind_speed_kmph`, `soil_moisture_pct`, `prior_regional_incidence_pct`) → `StandardScaler`
4. **Train/test split** — stratified split, holding out a test set for final, unbiased evaluation.
5. **Leakage control** — `risk_score` (the numeric value the target was derived from) was **excluded from the feature set**, since including it would let the model trivially recover the label rather than learn from genuine environmental signal.
6. **Model training** — three foundational classifiers trained inside the same pipeline: Logistic Regression, Decision Tree, K-Nearest Neighbors.
7. **Model selection** — compared on the test set by macro F1-score; the best model was re-evaluated and saved for deployment.
8. **Deployment** — the winning pipeline (preprocessing + classifier) serialized with `joblib` and served through a Streamlit application.

---

## 6. Exploratory Data Analysis

- **Data quality:** no missing values, no duplicate rows across all 10,000 records.
- **Distributions:** numerical features (temperature, humidity, rainfall, leaf wetness, wind speed, soil moisture, prior incidence) were visualized with histograms; no invalid ranges were found.
- **Relationships:** cross-tabulating disease against risk category showed clear differences in risk profile by disease — for example, **Powdery Mildew** observations were classified High-risk far more often (29.75%) than **Blast** (11.15%), reflecting each disease's distinct environmental sensitivity.
- **Correlation:** a correlation heatmap across numerical features and `risk_score` was used to check for strong linear relationships and confirm no unexpected collinearity among the environmental readings.
- **Outlier analysis:** boxplots were generated across all eight numerical features to check for extreme values before modeling.

---

## 7. Model Development & Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **K-Nearest Neighbors** | **69.15%** | **69.09%** | **67.35%** | **68.10%** |
| Logistic Regression | 68.60% | 69.62% | 64.15% | 65.99% |
| Decision Tree | 67.25% | 67.67% | 65.40% | 66.31% |

**KNN was selected as the deployed model**, achieving the highest macro F1-score. Its final, per-class test performance:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| High | 0.68 | 0.47 | 0.56 | 351 |
| Low | 0.77 | 0.70 | 0.73 | 715 |
| Medium | 0.64 | 0.76 | 0.69 | 934 |

*(Logistic Regression's per-class report shown as reference above; KNN's full classification report and confusion matrix are available in the notebook.)*

All three models performed similarly (within ~2 percentage points of each other on macro F1), suggesting the environmental features carry a genuine but moderate signal for risk classification, rather than one model architecture having a decisive advantage over the others on this data.

---

## 8. Discussion

The **High-risk** class was the hardest to recall correctly across all three models — it is also the least represented class (17.6% of the data), which is the expected effect of class imbalance on recall. KNN's non-parametric, distance-based approach appears to handle the mixed categorical/numerical feature space (after encoding) slightly better than the other two on this dataset, edging out Logistic Regression and Decision Tree on macro F1 despite Logistic Regression having marginally higher macro precision.

The inclusion of `growth_stage` alongside `day_of_year` is a deliberate but redundant design choice (see Section 4) — kept because it gave certain models an additional categorical split point, but it does not represent an independent source of information.

---

## 9. Limitations

- **Dataset origin is confirmed synthetic but not yet cited** — Section 4.2 confirms the data is formula-generated (threshold-derived target, perfectly balanced categories); the actual source/author still needs to be named before submission.
- **Feature redundancy** — `growth_stage` is fully determined by `day_of_year`.
- **No ROC-AUC** — the project brief marks ROC-AUC as applicable "where technically appropriate"; for this three-class problem it was not computed, and Accuracy/Precision/Recall/F1/Confusion Matrix were used instead as the primary evaluation set.
- **Class imbalance** — High-risk (17.6% of data) is under-represented, which visibly reduces its recall relative to Low and Medium.
- **Reproducibility environment** — the saved model was serialized with `scikit-learn 1.9.1`; loading it with a different (older or newer) `scikit-learn` version can raise compatibility warnings or errors, so the exact version should be pinned in `requirements.txt`.
- **Prototype scope** — this is an educational deployment; it has not been validated against real field outcomes and should not be used for actual crop-management decisions.

---

## 10. Future Scope

- Confirm and formally cite the dataset source (or, if synthetic, publish the generation methodology).
- Address class imbalance directly (class weighting, resampling) to improve High-risk recall.
- Add ROC-AUC (one-vs-rest) as a supplementary metric for a fuller picture of class separability.
- Extend beyond foundational models to ensemble methods (Random Forest, Gradient Boosting) once the project scope allows deep-learning-adjacent complexity.
- Replace the synthetic-looking dataset with real sensor/field-collected data if the project moves beyond prototype stage.

---

## 11. Conclusion

This project delivered a complete, reproducible pipeline for crop disease risk classification within a 10-day, zero-cost constraint: a validated dataset, thorough EDA, three foundational models trained and fairly compared, a defensible model-selection decision (KNN, macro F1 = 0.681), and a working, interactive Streamlit prototype that surfaces both the prediction and its full probability breakdown. The main outstanding item before formal submission is confirming and citing the dataset's true source.

---
## 12. References

1. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
2. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference.
3. Streamlit Inc. *Streamlit Documentation.* https://docs.streamlit.io
4. Plotly Technologies Inc. *Plotly Python Graphing Library.* https://plotly.com/python/
5. Dataset: *blob:https://github.com/39bb750c-f292-470c-a58b-f84e412dc272*