# An Explainable AIOps for Predictive IT Infrastructure Monitoring

**Case Study 21CSC601T — M.Tech AI, SRM Institute of Science and Technology**
**Author:** Akilan R (RA2312026010022)

---

## Overview

End-to-end AIOps pipeline that combines multi-model anomaly prediction,
SHAP-based explainability, LLM root-cause analysis, and an early-warning
precursor test on simulated IT infrastructure metrics.

### Research Questions
| RQ | Question | Key Finding |
|---|---|---|
| RQ1 | Which model best predicts incidents? | XGBoost F1 = 0.971 on held-out set |
| RQ2 | Can the system warn *before* the incident? | Lead time median 4.2 min, recognition ≠ forecasting |
| RQ3 | Are SHAP explanations faithful? | Spearman ρ = 0.981 vs permutation importance |
| RQ4 | Does LLM evidence improve RCA quality? | −24% hallucination rate with SHAP context |

---

## Repository Structure

```
AIOps/
├── pipeline/
│   ├── 01_simulate_data.py       IT infrastructure metric simulator (400 days)
│   ├── 02_feature_engineering.py 52-feature extraction with rolling windows
│   ├── 03_train_tabular.py       XGBoost + RF + SVM with Optuna HPO
│   ├── 04_train_sequence.py      LSTM / GRU / PatchTST sequence models
│   ├── 05_explain_shap.py        SHAP faithfulness evaluation + plots
│   ├── 06_llm_root_cause.py      LLaMA-3.2-3B ablation study
│   └── 07_early_warning.py       Lead-time / precursor test
├── dashboard/
│   ├── build_dashboard_v3.py     Dashboard builder (embeds all result figures)
│   └── dashboard_v3.html         Live interactive HTML dashboard
├── figures/                      All 25 result PNG figures
└── README.md
```

---

## Quick Start

```bash
pip install numpy pandas scikit-learn xgboost lightgbm optuna shap             torch transformers matplotlib seaborn

# 1. Generate synthetic data
python pipeline/01_simulate_data.py

# 2. Engineer features
python pipeline/02_feature_engineering.py

# 3. Train tabular models (XGBoost best)
python pipeline/03_train_tabular.py

# 4. Train sequence models
python pipeline/04_train_sequence.py

# 5. SHAP faithfulness evaluation
python pipeline/05_explain_shap.py

# 6. LLM root-cause ablation
python pipeline/06_llm_root_cause.py

# 7. Early-warning / precursor test
python pipeline/07_early_warning.py

# 8. Rebuild dashboard
python dashboard/build_dashboard_v3.py
```

---

## Key Results

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **XGBoost** | **0.963** | **0.979** | **0.971** | **0.997** |
| Random Forest | 0.941 | 0.958 | 0.949 | 0.991 |
| SVM | 0.887 | 0.923 | 0.905 | 0.974 |
| LSTM | 0.891 | 0.912 | 0.901 | 0.969 |
| GRU | 0.878 | 0.905 | 0.891 | 0.963 |
| PatchTST | 0.902 | 0.931 | 0.916 | 0.978 |

SHAP top features: `cpu_mean_30m`, `mem_std_30m`, `cpu_x_mem`, `disk_io_delta5`, `net_rx_mean_15m`

---

## Dashboard

Open `dashboard/dashboard_v3.html` in any browser — no server required.
Tabs: Overview · Dataset & Features · Hyperparameter Tuning · Training ·
Results · SHAP · Early-Warning · LLM & Logs · Key Scripts
