"""
07_early_warning.py
Evaluates whether the XGBoost model provides a genuine EARLY WARNING
(fires before the incident window) vs mere recognition (fires during it).

Precursor test:
  1. For each true-anomaly window, slide a look-ahead probe backwards
     in time and record when the model first crosses 0.5 probability.
  2. Lead time = ticks between first crossing and anomaly onset.
  3. Compare probability curve to a persistence baseline (P(t) = last label).
Input:  data/features.parquet,  data/raw_metrics.parquet,  models/xgb_best.pkl
Output: figures/fig_lead_time_curve.png  figures/fig_persistence_vs_model.png
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path

IN_FEAT  = Path('data/features.parquet')
IN_RAW   = Path('data/raw_metrics.parquet')
IN_MODEL = Path('models/xgb_best.pkl')
Path('figures').mkdir(exist_ok=True)

LOOK_BACK   = 60    # steps (30 min) to probe before each anomaly onset
THRESHOLD   = 0.50
FREQ_SEC    = 30
MIN_LEAD    = 2     # steps = 1 min minimum to count as genuine early warning


def find_onsets(y, min_gap=120):
    onsets = []
    in_anom = False
    for i, v in enumerate(y):
        if v == 1 and not in_anom:
            onsets.append(i)
            in_anom = True
        elif v == 0:
            in_anom = False
    # keep only onsets with enough look-back data
    return [o for o in onsets if o >= LOOK_BACK + 5]


def compute_lead_times(X, y, model):
    probs   = model.predict_proba(X)[:, 1]
    onsets  = find_onsets(y)
    leads   = []
    for t0 in onsets:
        # scan backwards from onset
        fired = None
        for k in range(LOOK_BACK, 0, -1):
            if probs[t0 - k] >= THRESHOLD:
                fired = t0 - k
                break
        if fired is not None:
            lead = t0 - fired
            if lead >= MIN_LEAD:
                leads.append(lead * FREQ_SEC / 60)   # minutes
    return np.array(leads), probs, onsets


def main():
    df    = pd.read_parquet(IN_FEAT)
    X     = df.drop(columns=['anomaly']).values.astype(np.float32)
    y     = df['anomaly'].values

    n_te  = int(len(X) * 0.15)
    X_te  = X[-n_te:]
    y_te  = y[-n_te:]

    model = joblib.load(IN_MODEL)
    leads, probs, onsets = compute_lead_times(X_te, y_te, model)

    print(f'Anomaly onsets evaluated : {len(onsets)}')
    print(f'Genuine early warnings   : {len(leads)} '
          f'({100*len(leads)/max(len(onsets),1):.0f}%)')
    print(f'Lead time  median={np.median(leads):.1f} min  '
          f'p25={np.percentile(leads,25):.1f}  p75={np.percentile(leads,75):.1f}')

    # Lead-time distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(leads, bins=20, color='#0055cc', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(leads), color='red', lw=1.5, linestyle='--',
               label=f'Median {np.median(leads):.1f} min')
    ax.set_xlabel('Lead Time (minutes)')
    ax.set_ylabel('Count')
    ax.set_title('Early-Warning Lead Time Distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/fig_lead_time_curve.png', dpi=150)
    plt.close(fig)

    # Persistence baseline comparison
    persistence = np.roll(y_te.astype(float), 1)
    persistence[0] = 0
    auc_model = roc_auc_score(y_te, probs)
    auc_pers  = roc_auc_score(y_te, persistence)
    print(f'AUC  model={auc_model:.3f}  persistence={auc_pers:.3f}  '
          f'(delta={auc_model-auc_pers:+.3f})')

    # Plot model vs persistence probability for first 2000 test steps
    fig, ax = plt.subplots(figsize=(10, 3))
    window = 2000
    t = np.arange(window)
    ax.plot(t, probs[:window], lw=0.8, label='XGBoost P(anomaly)')
    ax.plot(t, persistence[:window], lw=0.8, linestyle='--',
            color='orange', label='Persistence baseline')
    ax.fill_between(t, 0, y_te[:window].astype(float)*0.9,
                    alpha=0.15, color='red', label='True anomaly')
    ax.axhline(THRESHOLD, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Test Step (×30 s)')
    ax.set_ylabel('Probability')
    ax.set_title('Model vs Persistence — 2000-step window')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/fig_persistence_vs_model.png', dpi=150)
    plt.close(fig)

    print('Figures saved → figures/')


if __name__ == '__main__':
    main()
