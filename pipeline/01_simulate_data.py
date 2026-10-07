"""
01_simulate_data.py
Simulate 400 days of IT infrastructure metrics at 30-second intervals.
Injects 7 anomaly types: cpu_spike, mem_leak, disk_thrash, net_storm,
cascade, k8s_oom, db_lock.
Outputs: data/raw_metrics.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path('data/raw_metrics.parquet')

METRICS = ['cpu', 'mem', 'disk_io', 'net_rx', 'net_tx']
ANOMALY_TYPES = [
    'cpu_spike', 'mem_leak', 'disk_thrash',
    'net_storm', 'cascade', 'k8s_oom', 'db_lock'
]


def inject_anomaly(df, t0, atype, rng):
    dur = {'cpu_spike': 20, 'mem_leak': 120, 'disk_thrash': 40,
           'net_storm': 30, 'cascade': 60, 'k8s_oom': 90, 'db_lock': 50}
    d = dur.get(atype, 30)
    end = min(t0 + d, len(df))
    if atype == 'cpu_spike':
        df.loc[df.index[t0:end], 'cpu'] += rng.uniform(40, 60, end - t0)
    elif atype == 'mem_leak':
        leak = np.linspace(0, 40, end - t0)
        df.loc[df.index[t0:end], 'mem'] += leak
    elif atype == 'disk_thrash':
        df.loc[df.index[t0:end], 'disk_io'] += rng.uniform(50, 80, end - t0)
    elif atype == 'net_storm':
        df.loc[df.index[t0:end], 'net_rx'] *= rng.uniform(3, 8)
        df.loc[df.index[t0:end], 'net_tx'] *= rng.uniform(2, 5)
    elif atype == 'cascade':
        df.loc[df.index[t0:end], 'cpu']    += rng.uniform(30, 50, end - t0)
        df.loc[df.index[t0:end], 'mem']    += rng.uniform(20, 40, end - t0)
        df.loc[df.index[t0:end], 'disk_io']+= rng.uniform(20, 40, end - t0)
    elif atype == 'k8s_oom':
        df.loc[df.index[t0:end], 'mem'] = np.clip(
            df.loc[df.index[t0:end], 'mem'] + rng.uniform(45, 60, end - t0), 0, 100)
    elif atype == 'db_lock':
        df.loc[df.index[t0:end], 'disk_io'] += rng.uniform(60, 90, end - t0)
        df.loc[df.index[t0:end], 'cpu']     += rng.uniform(10, 25, end - t0)
    df['anomaly'].iloc[t0:end] = 1
    df['anomaly_type'].iloc[t0:end] = atype


def simulate_infrastructure(n_days=400, freq_sec=30, seed=42):
    rng = np.random.default_rng(seed)
    n   = n_days * (86400 // freq_sec)
    idx = pd.date_range('2024-01-01', periods=n, freq=f'{freq_sec}s')
    df  = pd.DataFrame(index=idx)

    day_cycle = np.sin(2 * np.pi * np.arange(n) / (86400 // freq_sec))
    for m in METRICS:
        base  = 30 + 20 * day_cycle
        noise = rng.normal(0, 1, n).cumsum() * 0.005
        df[m] = np.clip(base + noise, 1, 99)

    df['anomaly']      = 0
    df['anomaly_type'] = 'normal'

    n_events = int(n_days * 3)
    for _ in range(n_events):
        t0    = rng.integers(120, n - 200)
        atype = rng.choice(ANOMALY_TYPES)
        inject_anomaly(df, t0, atype, rng)

    for m in METRICS:
        df[m] = df[m].clip(0, 100)

    print(f'Generated {len(df):,} rows | anomaly rate: {df["anomaly"].mean():.1%}')
    print(f'Anomaly type counts:\n{df[df.anomaly==1]["anomaly_type"].value_counts()}')
    return df


if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    df = simulate_infrastructure()
    df.to_parquet(OUT)
    print(f'Saved → {OUT}')
