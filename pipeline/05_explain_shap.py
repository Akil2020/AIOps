"""
05_explain_shap.py
Compute SHAP values for the XGBoost model.
Evaluates faithfulness via Spearman correlation with permutation importance.
Generates global bar, beeswarm, and per-prediction waterfall plots.
Input:  data/features.parquet, models/xgb_best.pkl
Output: figures/fig_shap_*.png
"""
import numpy as np
import pandas as pd
import joblib, shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from pathlib import Path

IN_DATA  = Path('data/features.parquet')
IN_MODEL = Path('models/xgb_best.pkl')
Path('figures').mkdir(exist_ok=True)

SEED = 42
N_EXPLAIN = 2000


def main():
    df    = pd.read_parquet(IN_DATA)
    X     = df.drop(columns=['anomaly'])
    y     = df['anomaly'].values
    feat  = X.columns.tolist()

    n_te  = int(len(X) * 0.15)
    X_te  = X.values[-n_te:].astype(np.float32)
    y_te  = y[-n_te:]

    model = joblib.load(IN_MODEL)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    idx = np.random.default_rng(SEED).choice(len(X_te), N_EXPLAIN, replace=False)
    Xs  = X_te[idx]
    sv  = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]   # binary: take positive class

    shap_rank = np.argsort(-np.abs(sv).mean(0))

    # Permutation importance (faithfulness baseline)
    perm = permutation_importance(model, X_te, y_te,
                                  n_repeats=10, scoring='f1', random_state=SEED)
    perm_rank = np.argsort(-perm.importances_mean)
    rho, pval = spearmanr(shap_rank, perm_rank)
    print(f'Faithfulness  rho={rho:.3f}  p={pval:.2e}')

    # Global bar chart
    shap.summary_plot(sv, Xs, feature_names=feat, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('figures/fig_shap_global_incident.png', dpi=150)
    plt.close()

    # Beeswarm
    shap.summary_plot(sv, Xs, feature_names=feat, show=False)
    plt.tight_layout()
    plt.savefig('figures/fig_shap_beeswarm_incident.png', dpi=150)
    plt.close()

    # Waterfall plots for 2 anomaly samples
    exp = shap.Explanation(values=sv, base_values=explainer.expected_value,
                           data=Xs, feature_names=feat)
    for k, sample_idx in enumerate([0, 1]):
        shap.plots.waterfall(exp[sample_idx], show=False)
        plt.tight_layout()
        plt.savefig(f'figures/fig_shap_waterfall_{k}.png', dpi=150)
        plt.close()
        print(f'Top-3 features for sample {k}: ' +
              ', '.join([feat[i] for i in np.argsort(-np.abs(sv[sample_idx]))[:3]]))

    print('Done — figures saved to figures/')


if __name__ == '__main__':
    main()
