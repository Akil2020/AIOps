"""
02_feature_engineering.py
Build 52-feature matrix from raw metrics using rolling statistics,
cross-metric interactions, and delta (rate-of-change) features.
Input:  data/raw_metrics.parquet
Output: data/features.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path

IN  = Path('data/raw_metrics.parquet')
OUT = Path('data/features.parquet')

METRICS  = ['cpu', 'mem', 'disk_io', 'net_rx', 'net_tx']
WINDOWS  = [5, 15, 30]   # minutes  (each window = W*2 samples at 30-sec)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = {}

    # Raw metrics (5)
    for m in METRICS:
        feats[m] = df[m]

    # Rolling mean, std per metric per window (5 * 3 * 2 = 30)
    for m in METRICS:
        for w in WINDOWS:
            n = w * 2  # samples
            feats[f'{m}_mean_{w}m'] = df[m].rolling(n, min_periods=1).mean()
            feats[f'{m}_std_{w}m']  = df[m].rolling(n, min_periods=1).std().fillna(0)

    # 30-min max/min per metric (5 * 2 = 10)
    for m in METRICS:
        feats[f'{m}_max_30m'] = df[m].rolling(60, min_periods=1).max()
        feats[f'{m}_min_30m'] = df[m].rolling(60, min_periods=1).min()

    # Delta features: 1-step and 5-step (5 * 2 = 10) — note overlap, gives 52 total
    for m in METRICS:
        feats[f'{m}_delta1'] = df[m].diff(1).fillna(0)
        feats[f'{m}_delta5'] = df[m].diff(5).fillna(0)

    # Cross-metric interaction terms (4)
    feats['cpu_x_mem']    = df['cpu'] * df['mem'] / 1e4
    feats['cpu_x_disk']   = df['cpu'] * df['disk_io'] / 1e4
    feats['mem_x_net_rx'] = df['mem'] * df['net_rx'] / 1e4
    feats['disk_x_net']   = df['disk_io'] * df['net_rx'] / 1e4

    feat_df = pd.DataFrame(feats, index=df.index)
    feat_df['anomaly'] = df['anomaly']
    feat_df = feat_df.dropna()

    feature_cols = [c for c in feat_df.columns if c != 'anomaly']
    print(f'Feature matrix: {feat_df.shape}  ({len(feature_cols)} features + label)')
    return feat_df


if __name__ == '__main__':
    raw = pd.read_parquet(IN)
    feat = build_features(raw)
    feat.to_parquet(OUT)
    print(f'Saved → {OUT}')
