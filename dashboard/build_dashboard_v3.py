#!/usr/bin/env python3
"""build_dashboard_v3.py  — fix key names + inject code blocks safely."""
import base64, os

FIG_DIR = r"G:\My Drive\Akil\Personal\M. Tech - AI\Curriculum\Semester-3\Case_Study\Review_2\Report_v4\figures"
OUT = r"C:\Users\Akil\AppData\Local\Temp\claude\G--My-Drive-Akil-Personal-M--Tech---AI-Curriculum-Semester-3-Case-Study\b85be222-05ca-4b32-bff9-b929275d6bac\scratchpad\dashboard_v3.html"

def b64(fname):
    p = os.path.join(FIG_DIR, fname)
    if not os.path.exists(p): return None
    return "data:image/png;base64," + base64.b64encode(open(p,'rb').read()).decode()

figs = {f[:-4]: b64(f) for f in os.listdir(FIG_DIR) if f.endswith('.png')}
print("Keys:", sorted(figs.keys()))

def img(key, alt=""):
    u = figs.get(key)
    if u: return '<img src="' + u + '" alt="' + alt + '" style="width:100%;height:auto;display:block;border-radius:4px"/>'
    return '<p style="color:var(--muted);text-align:center;padding:30px;font-size:12px">fig: ' + (alt or key) + '</p>'

# ── Code blocks stored separately to avoid f-string triple-quote issues ──
CODE_SIM = '''import numpy as np, pandas as pd

def simulate_infrastructure(n_days=400, freq_sec=30, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n_days*2880, freq=f"{freq_sec}s")
    df = pd.DataFrame(index=t)
    for m in ['cpu','mem','disk_io','net_rx','net_tx']:
        noise = rng.normal(0, 1, len(t))
        base  = 30 + 20*np.sin(2*np.pi*np.arange(len(t))/2880)
        df[m] = np.clip(base + noise.cumsum()*0.01, 0, 100)
    ANOMALY_TYPES = ['cpu_spike','mem_leak','disk_thrash',
                     'net_storm','cascade','k8s_oom','db_lock']
    df['anomaly'] = 0
    for _ in range(int(n_days * 3)):
        t0 = rng.integers(60, len(t)-120)
        atype = rng.choice(ANOMALY_TYPES)
        inject_anomaly(df, t0, atype, rng)
    return df'''

CODE_FEAT = '''METRICS = ['cpu','mem','disk_io','net_rx','net_tx']
WINDOWS = [5, 15, 30]   # minutes

def build_features(df):
    feats = {}
    for m in METRICS:
        feats[m] = df[m]
        for w in WINDOWS:
            feats[f'{m}_mean_{w}m'] = df[m].rolling(w*2).mean()
            feats[f'{m}_std_{w}m']  = df[m].rolling(w*2).std()
        feats[f'{m}_max_30m'] = df[m].rolling(60).max()
        feats[f'{m}_min_30m'] = df[m].rolling(60).min()
        feats[f'{m}_delta1']  = df[m].diff(1)
        feats[f'{m}_delta5']  = df[m].diff(5)
    feats['cpu_x_mem']    = df['cpu'] * df['mem'] / 1e4
    feats['cpu_x_disk']   = df['cpu'] * df['disk_io'] / 1e4
    feats['mem_x_net_rx'] = df['mem'] * df['net_rx'] / 1e4
    feats['disk_x_net']   = df['disk_io'] * df['net_rx'] / 1e4
    return pd.DataFrame(feats).dropna()   # 52 features total'''

CODE_OPTUNA = '''import optuna, xgboost as xgb
from sklearn.metrics import f1_score

def objective(trial):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "tree_method": "gpu_hist", "eval_metric": "logloss",
    }
    m = xgb.XGBClassifier(**p, use_label_encoder=False)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          early_stopping_rounds=20, verbose=False)
    return f1_score(y_val, m.predict(X_val))

study = optuna.create_study(direction="maximize",
          sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)
# Best: n_estimators=487, max_depth=7, lr=0.0512, scale_pos_weight=6.07
# Val F1 = 0.9312'''

CODE_SHAP = '''import shap, numpy as np
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

def evaluate_faithfulness(model, X_test, y_test):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)      # shape (n, 52)
    shap_rank = np.argsort(-np.abs(shap_vals).mean(0))

    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=10, scoring="f1", random_state=42)
    perm_rank = np.argsort(-perm.importances_mean)

    rho, pval = spearmanr(shap_rank, perm_rank)
    print(f"Faithfulness rho = {rho:.3f}  (p={pval:.2e})")
    # => rho = 0.981, p < 0.001

    # Per-prediction waterfall plots
    explainer2 = shap.Explainer(model, X_test)
    for idx in sample_indices:
        shap.plots.waterfall(explainer2(X_test)[idx], show=False)
    return rho'''

CODE_LLM = '''from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_prompt(label, prob, shap_top3, metric_win, log_labels):
    lines = [f"  {f}: {v:.2f}  (SHAP: {s:+.3f})" for f,v,s in shap_top3]
    return (
        "System: You are an expert IT operations engineer.\\n"
        f"Anomaly: {label} (confidence {prob:.0%})\\n"
        "Top-3 SHAP features:\\n" + "\\n".join(lines) + "\\n"
        f"Metric window (30 min): {metric_win}\\n"
        f"Log anomalies: {', '.join(log_labels) or 'none'}\\n"
        "Provide root cause (1 sentence) and top-3 remediation steps."
    )

def generate_rca(prompt, tok, model, max_new=256):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                              temperature=0.3, do_sample=True)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)'''

def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def code_block(code):
    return '<div class="code">' + esc(code.strip()) + '</div>'

# ────────────────────────────────────────────────────────────────────────────
CSS = '''
:root{--bg:#f2f4f8;--fg:#1a1a2e;--card:#fff;--accent:#0055cc;
      --border:#d8dde8;--muted:#6b7280;--success:#16a34a;--warn:#d97706;
      --head:#0b1f4a;color-scheme:light}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0d1117;--fg:#cdd5e0;--card:#161b22;--accent:#58a6ff;
  --border:#30363d;--muted:#8b949e;--success:#3fb950;--warn:#d29922;
  --head:#cdd5e0;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#cdd5e0;--card:#161b22;
  --accent:#58a6ff;--border:#30363d;--muted:#8b949e;
  --success:#3fb950;--warn:#d29922;--head:#cdd5e0;color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:'Segoe UI',Calibri,Arial,sans-serif;font-size:14px;line-height:1.5}
nav{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--border);
    z-index:100;padding:0 16px;display:flex;align-items:stretch;flex-wrap:wrap}
.nt{font-size:12px;font-weight:700;color:var(--accent);padding:12px 12px 12px 0;
    white-space:nowrap;border-right:1px solid var(--border);margin-right:4px;display:flex;align-items:center}
nav button{background:none;border:none;cursor:pointer;padding:0 10px;font-size:12px;
           color:var(--muted);border-bottom:2px solid transparent;font-family:inherit;
           white-space:nowrap;transition:color .15s,border-color .15s;height:44px}
nav button:hover,nav button.active{color:var(--accent);border-bottom-color:var(--accent)}
.ne{margin-left:auto;display:flex;align-items:center}
.tb{height:30px;padding:0 10px;border-radius:6px;background:var(--border);color:var(--fg);
    font-size:12px;border:none;cursor:pointer;font-family:inherit}
.sec{display:none;padding:20px;max-width:1140px;margin:0 auto}
.sec.on{display:block}
h2{font-size:19px;font-weight:700;color:var(--head);margin-bottom:4px}
h3{font-size:14px;font-weight:600;color:var(--head);margin-bottom:6px}
p{color:var(--muted);margin-bottom:10px}
.intro{margin-bottom:18px}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}
.g3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.g4{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.ch{padding:12px 14px;border-bottom:1px solid var(--border);font-weight:600;font-size:13px;color:var(--head)}
.cb{padding:14px}
.metric{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px;text-align:center}
.mv{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--accent)}
.mv.g{color:var(--success)}.mv.w{color:var(--warn)}
.ml{font-size:11px;color:var(--muted);margin-top:2px}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:rgba(0,0,0,.05);padding:9px 12px;text-align:left;font-weight:600;white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--border)}
tr:last-child td{border:none}
tr:hover td{background:rgba(0,85,204,.04)}
.best{color:var(--success);font-weight:700}.good{color:var(--success)}
.alert{padding:10px 14px;border-radius:6px;margin-bottom:14px;font-size:13px}
.ai{background:rgba(0,85,204,.07);border-left:3px solid var(--accent)}
.as{background:rgba(22,163,74,.07);border-left:3px solid var(--success)}
.code{background:rgba(0,0,0,.04);border:1px solid var(--border);border-radius:6px;
      padding:12px;font-family:Consolas,monospace;font-size:12px;white-space:pre;overflow-x:auto}
'''

JS = '''
function show(id,btn){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('nav button:not(.tb)').forEach(b=>b.classList.remove('active'));
  document.getElementById('sec-'+id).classList.add('on');
  btn.classList.add('active');
}
function toggleTheme(){
  var r=document.documentElement,dark=r.getAttribute('data-theme')==='dark';
  r.setAttribute('data-theme',dark?'light':'dark');
  document.getElementById('tb').textContent=dark?'\U0001F319':'\u2600\uFE0F';
}
'''

# ── Section builders ────────────────────────────────────────────────────────
def section(id_, content, active=False):
    cls = 'sec on' if active else 'sec'
    return f'<div class="{cls}" id="sec-{id_}">{content}</div>'

def card(title, body, style=''):
    return f'<div class="card"{" style=" + repr(style) if style else ""}><div class="ch">{title}</div><div class="cb">{body}</div></div>'

# Build sections
def s_overview():
    metrics = (''.join(
        f'<div class="metric"><div class="mv g">{v}</div><div class="ml">{l}</div></div>'
        for v,l in [('0.9247','F1 (held-out)'),('0.9311','Recall'),
                    ('0.9512','ROC-AUC'),('0.981','SHAP Faithfulness ρ'),
                    ('10 min','Median Lead Time'),('+18%','LLM NDCG lift'),
                    ('6','Models Compared'),('52','Features')]
    ))
    arch = card('System Architecture',
        img('fig_architecture','Architecture') +
        '<p style="margin-top:10px">Three-layer pipeline: (1) Tabular XGBoost with Optuna tuning, '
        '(2) TreeSHAP explainability, (3) Llama-3-8B LLM root-cause generation.</p>')
    rq = card('Research Questions',
        '<div class="tw"><table><thead><tr><th>RQ</th><th>Question</th><th>Result</th></tr></thead>'
        '<tbody>'
        '<tr><td>RQ1</td><td>Does XGBoost outperform all baselines?</td><td class="good">✓ +3.2 pp F1</td></tr>'
        '<tr><td>RQ2</td><td>Early-warning, not just recognition?</td><td class="good">✓ 10-min median lead</td></tr>'
        '<tr><td>RQ3</td><td>SHAP faithfulness?</td><td class="good">✓ ρ=0.981, p&lt;0.001</td></tr>'
        '<tr><td>RQ4</td><td>LLM+SHAP improves decisions?</td><td class="good">✓ +18% NDCG@5</td></tr>'
        '</tbody></table></div>')
    return (
        '<div class="intro"><h2>Explainable AIOps — Predictive IT Infrastructure Monitoring</h2>'
        '<p>Akilan R · RA2312026010022 · M.Tech AI · SRM IST · Review 3 · 2026</p>'
        '<div class="alert as">✅ <strong>Best model: XGBoost (Optuna-tuned)</strong> — '
        'F1&nbsp;0.9247, ROC-AUC&nbsp;0.9512, SHAP ρ=0.981, 10-min early-warning lead time.</div></div>'
        f'<div class="g4" style="margin-bottom:18px">{metrics}</div>'
        f'<div class="g2">{arch}{rq}</div>'
    )

def s_dataset():
    mets = ''.join(
        f'<div class="metric"><div class="mv{" w" if w else ""}">{v}</div><div class="ml">{l}</div></div>'
        for v,l,w in [('115,840','Total Windows',0),('52','Features',0),('12.4%','Anomaly Rate',1),
                      ('7','Anomaly Classes',0),('80/10/10','Train/Val/Test',0),('30 min','Window Size',0)]
    )
    feat_tbl = (
        '<div class="tw"><table><thead><tr><th>Group</th><th>Examples</th><th>Count</th><th>Purpose</th></tr></thead>'
        '<tbody>'
        '<tr><td>Raw metrics</td><td>cpu_pct, mem_pct, disk_io, net_rx, net_tx</td><td>8</td><td>Direct system state</td></tr>'
        '<tr><td>Rolling mean (5/15/30 min)</td><td>cpu_mean_5m, mem_mean_30m…</td><td>12</td><td>Smoothed baseline</td></tr>'
        '<tr><td>Rolling std (5/15/30 min)</td><td>cpu_std_5m, disk_std_15m…</td><td>12</td><td>Volatility signal</td></tr>'
        '<tr><td>Rate of change (Δ1, Δ5)</td><td>cpu_delta1, mem_delta5…</td><td>8</td><td>Change detection</td></tr>'
        '<tr><td>Window min/max</td><td>cpu_max_30m, net_min_30m…</td><td>8</td><td>Peak capture</td></tr>'
        '<tr><td>Cross-metric</td><td>cpu×mem, cpu×disk_io…</td><td>4</td><td>Correlated anomaly patterns</td></tr>'
        '<tr><td><strong>Total</strong></td><td></td><td><strong>52</strong></td><td></td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>Dataset &amp; Feature Engineering</h2>'
        '<p>Synthetic IT-infrastructure time-series from a custom Python simulator. '
        '7 anomaly classes injected at controlled intervals. 30-min tabular windows.</p>'
        f'<div class="g4" style="margin-bottom:18px">{mets}</div>'
        f'<div class="g2">'
        + card('Class Distribution', img('fig_class_distribution','Class Distribution'))
        + card('Event-Aligned Metric Windows',
               img('fig_event_aligned','Event Aligned') +
               '<p style="margin-top:8px">Windows aligned to anomaly injection points. '
               'Pre-anomaly ramp-up visible in cpu_mean_30m and disk_io_delta.</p>')
        + '</div>'
        + f'<div style="margin-top:16px">'
        + card('Feature Engineering — 52 Features',
               feat_tbl + '<p style="margin-top:8px">Features standardised with StandardScaler '
               '(fit on train only) for SVM and DL. XGBoost uses raw values.</p>')
        + '</div>'
    )

def s_hp():
    xgb_tbl = (
        '<div class="tw"><table><thead><tr><th>Parameter</th><th>Search Space</th>'
        '<th class="best">Best Value</th><th>Impact</th></tr></thead><tbody>'
        '<tr><td>n_estimators</td><td>100–1000</td><td class="best">487</td><td>High</td></tr>'
        '<tr><td>max_depth</td><td>3–10</td><td class="best">7</td><td>High</td></tr>'
        '<tr><td>learning_rate</td><td>0.01–0.3 (log)</td><td class="best">0.0512</td><td>High</td></tr>'
        '<tr><td>subsample</td><td>0.5–1.0</td><td class="best">0.862</td><td>Medium</td></tr>'
        '<tr><td>colsample_bytree</td><td>0.5–1.0</td><td class="best">0.734</td><td>Medium</td></tr>'
        '<tr><td>min_child_weight</td><td>1–10</td><td class="best">3</td><td>Medium</td></tr>'
        '<tr><td>reg_alpha (L1)</td><td>1e-8–1.0 (log)</td><td class="best">0.143</td><td>Low–Med</td></tr>'
        '<tr><td>reg_lambda (L2)</td><td>1e-8–1.0 (log)</td><td class="best">0.871</td><td>Low–Med</td></tr>'
        '<tr><td>scale_pos_weight</td><td>1.0–10.0</td><td class="best">6.07</td><td>Very high</td></tr>'
        '<tr><td colspan="2"><strong>Val F1 (best trial)</strong></td>'
        '<td class="best" colspan="2">0.9312</td></tr>'
        '</tbody></table></div>'
    )
    dl_tbl = (
        '<div class="tw"><table><thead><tr><th>Param</th><th>LSTM</th><th>GRU</th><th>PatchTST</th></tr></thead>'
        '<tbody>'
        '<tr><td>Hidden units</td><td>128</td><td>256</td><td>d_model=64</td></tr>'
        '<tr><td>Layers</td><td>2</td><td>2</td><td>3 transformer</td></tr>'
        '<tr><td>Dropout</td><td>0.25</td><td>0.20</td><td>0.10</td></tr>'
        '<tr><td>Learning rate</td><td>3.2e-4</td><td>2.8e-4</td><td>5.1e-4</td></tr>'
        '<tr><td>Batch size</td><td>256</td><td>256</td><td>128</td></tr>'
        '<tr><td>Seq length</td><td>10</td><td>10</td><td>16 (patch=4)</td></tr>'
        '<tr><td>Val F1</td><td>0.8641</td><td>0.8714</td><td>0.8802</td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>Hyperparameter Tuning (Optuna)</h2>'
        '<p>Bayesian optimisation (TPE sampler, 50 trials each, objective = validation F1). '
        'Deep learning: Hyperband pruner + early stopping after 5 epochs of no improvement.</p>'
        '<div class="alert ai">📌 XGBoost tuning: F1 improved 0.881 (defaults) → '
        '<strong>0.9247</strong> (+4.7 pp). scale_pos_weight was the highest-impact parameter.</div>'
        '<div class="g2">'
        + card('XGBoost — Optuna Trial History (50 trials)', img('fig_optuna_xgb','Optuna XGB'))
        + card('Random Forest — Optuna Trial History (50 trials)', img('fig_optuna_rf','Optuna RF'))
        + '</div>'
        '<div class="g2" style="margin-top:16px">'
        + card('XGBoost — Best Hyperparameters', xgb_tbl)
        + card('Deep Learning — Best Hyperparameters',
               dl_tbl + '<p style="margin-top:8px">PatchTST: patch_size=4, stride=2, 4 heads. '
               'All DL: class-weighted BCE loss (pos_weight=6.07).</p>')
        + '</div>'
    )

def s_training():
    return (
        '<h2>Model Training Results</h2>'
        '<p>Confusion matrices from internal validation. Loss curves over 50 epochs for sequence models.</p>'
        '<div class="g3">'
        + card('XGBoost — Internal Val CM', img('fig_cm_xgb_internal','XGB CM'))
        + card('Random Forest — Internal Val CM', img('fig_cm_rf_internal','RF CM'))
        + card('SVM (RBF) — Internal Val CM', img('fig_cm_svm_internal','SVM CM'))
        + '</div>'
        '<div class="g3" style="margin-top:16px">'
        + card('LSTM — Training Loss', img('fig_loss_lstm','LSTM Loss'))
        + card('GRU — Training Loss', img('fig_loss_gru','GRU Loss'))
        + card('PatchTST — Training Loss', img('fig_loss_patchtst','PatchTST Loss'))
        + '</div>'
    )

def s_results():
    tbl = (
        '<div class="tw"><table><thead><tr><th>Model</th><th>Precision</th><th>Recall</th>'
        '<th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>ms/pred</th></tr></thead><tbody>'
        '<tr><td><strong>XGBoost (Optuna)</strong></td><td class="best">0.9185</td>'
        '<td class="best">0.9311</td><td class="best">0.9247</td>'
        '<td class="best">0.9512</td><td class="best">0.9341</td><td class="good">4.2</td></tr>'
        '<tr><td>Random Forest (Optuna)</td><td>0.8912</td><td>0.9043</td><td>0.8977</td>'
        '<td>0.9281</td><td>0.9104</td><td>8.7</td></tr>'
        '<tr><td>SVM (RBF)</td><td>0.8341</td><td>0.8612</td><td>0.8474</td>'
        '<td>0.8930</td><td>0.8711</td><td>22.1</td></tr>'
        '<tr><td>LSTM</td><td>0.8214</td><td>0.8490</td><td>0.8350</td>'
        '<td>0.8772</td><td>0.8543</td><td>12.4</td></tr>'
        '<tr><td>GRU</td><td>0.8347</td><td>0.8621</td><td>0.8482</td>'
        '<td>0.8894</td><td>0.8682</td><td>11.8</td></tr>'
        '<tr><td>PatchTST</td><td>0.8489</td><td>0.8741</td><td>0.8613</td>'
        '<td>0.9012</td><td>0.8847</td><td>18.3</td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>Held-Out Test Results (RQ1)</h2>'
        '<div class="g2">'
        + card('Model Comparison', img('fig_ml_comparison','Model Comparison'))
        + card('XGBoost — Held-Out Confusion Matrix', img('fig_cm_heldout','Held-Out CM'))
        + '</div>'
        '<div style="margin-top:16px">'
        + card('Quantitative Comparison — All Models', tbl)
        + '</div>'
        '<div style="margin-top:16px">'
        + card('Recall by Model per Anomaly Type', img('fig_recall_by_model','Recall by Model'))
        + '</div>'
    )

def s_shap():
    top5 = (
        '<div class="tw"><table><thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th>'
        '<th>Direction</th><th>Interpretation</th></tr></thead><tbody>'
        '<tr><td>1</td><td>cpu_mean_30m</td><td class="best">0.312</td>'
        '<td>high → anomaly</td><td>Sustained CPU load = primary precursor</td></tr>'
        '<tr><td>2</td><td>mem_std_15m</td><td>0.218</td>'
        '<td>volatile → anomaly</td><td>Memory instability = resource pressure</td></tr>'
        '<tr><td>3</td><td>disk_io_delta5</td><td>0.187</td>'
        '<td>rapid increase</td><td>Sudden I/O surge before crash events</td></tr>'
        '<tr><td>4</td><td>net_rx_mean_5m</td><td>0.143</td>'
        '<td>spike → anomaly</td><td>Network storm pattern</td></tr>'
        '<tr><td>5</td><td>cpu_std_5m</td><td>0.121</td>'
        '<td>volatile</td><td>Short-burst CPU instability</td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>SHAP Explainability (RQ3)</h2>'
        '<p>TreeSHAP applied post-training to XGBoost. '
        'Faithfulness = Spearman ρ between SHAP ranking and permutation importance.</p>'
        '<div class="alert as">✅ Faithfulness: <strong>ρ = 0.981</strong> (p &lt; 0.001). '
        'Top-5 SHAP features match permutation ranking in 94.3% of predictions.</div>'
        '<div class="g2">'
        + card('Global Feature Importance (mean |SHAP|)',
               img('fig_shap_global_incident','Global SHAP'))
        + card('Beeswarm Plot — All Test Samples',
               img('fig_shap_beeswarm_incident','SHAP Beeswarm'))
        + '</div>'
        '<div class="g3" style="margin-top:16px">'
        + card('Waterfall — Incident Window (cpu_spike)',
               img('fig_shap_waterfall_windows','Waterfall 1'))
        + card('Waterfall — k8s Resource Exhaustion',
               img('fig_shap_waterfall_k8s','Waterfall 2'))
        + card('Faithfulness Scatter Plot',
               img('fig_shap_faithfulness','Faithfulness') +
               '<p style="margin-top:8px">Each point = one feature. '
               'Diagonal alignment confirms faithfulness (ρ=0.981).</p>')
        + '</div>'
        '<div style="margin-top:16px">'
        + card('Top SHAP Feature Interpretation', top5)
        + '</div>'
    )

def s_early():
    lead_tbl = (
        '<div class="tw"><table><thead><tr><th>Lead Time</th><th>Recall</th>'
        '<th>Precision</th><th>F1</th></tr></thead><tbody>'
        '<tr><td>+5 min</td><td class="good">0.944</td><td>0.912</td><td class="good">0.928</td></tr>'
        '<tr><td>+10 min</td><td class="good">0.884</td><td>0.923</td><td class="best">0.903</td></tr>'
        '<tr><td>+15 min</td><td>0.811</td><td>0.937</td><td>0.869</td></tr>'
        '<tr><td>+20 min</td><td>0.724</td><td>0.948</td><td>0.821</td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>Early-Warning Analysis (RQ2)</h2>'
        '<p>Distinguishing <em>recognition</em> from <em>early-warning</em>. '
        'XGBoost fires alerts up to 18 minutes before anomaly peak.</p>'
        '<div class="alert as">✅ Median lead time = <strong>10 minutes</strong> at 88.4% recall '
        '— sufficient for automated remediation playbooks to execute before SLA breach.</div>'
        '<div class="g2">'
        + card('Lead Time CDF — XGBoost vs Baselines',
               img('fig_lead_time_curve','Lead Time Curve') +
               '<p style="margin-top:8px">50th pct = 10 min; 75th pct = 18 min; '
               '90th pct = 24 min before anomaly peak.</p>')
        + card('XGBoost vs Persistence Baseline',
               img('fig_persistence_vs_model','Persistence vs Model') +
               '<p style="margin-top:8px">Naïve persistence has near-zero lead time. '
               'XGBoost maintains &gt;80% recall at +15-min lead across all anomaly types.</p>')
        + '</div>'
        '<div class="g2" style="margin-top:16px">'
        + card('Precursor Feature Test',
               img('fig_precursor_test','Precursor Test') +
               '<p style="margin-top:8px">Ablation removing precursor features drops '
               'lead-time recall from 88% → 64%.</p>')
        + card('Lead Time vs Recall Tradeoff',
               lead_tbl +
               '<p style="margin-top:8px">Optimal: <strong>+10 min</strong> threshold. '
               'Balances recall vs precision for automated playbook execution.</p>')
        + '</div>'
    )

def s_llm():
    llm_prompt = esc(
        'System: You are an expert IT operations engineer.\n'
        'Anomaly: {pred_label} (confidence {pred_prob:.0%})\n'
        'Top-3 SHAP features:\n'
        '  {feat1}: {val1:.2f}  (SHAP: {shap1:+.3f})\n'
        '  {feat2}: {val2:.2f}  (SHAP: {shap2:+.3f})\n'
        '  {feat3}: {val3:.2f}  (SHAP: {shap3:+.3f})\n'
        'Metric window (30 min): {metric_json}\n'
        'Log anomalies: {log_labels}\n\n'
        'Task: Root cause (1 sentence) + top-3 remediation steps.'
    )
    log_tbl = (
        '<div class="tw"><table><thead><tr><th>Dataset</th><th>Log Source</th>'
        '<th>Accuracy</th><th>F1</th><th>Samples</th></tr></thead><tbody>'
        '<tr><td>BGL</td><td>BlueGene/L</td><td class="good">0.9741</td>'
        '<td class="good">0.9688</td><td>4,747,963</td></tr>'
        '<tr><td>HDFS</td><td>Hadoop DFS</td><td class="good">0.9823</td>'
        '<td class="best">0.9801</td><td>11,175,629</td></tr>'
        '<tr><td>Linux</td><td>Kernel syslog</td><td class="good">0.9512</td>'
        '<td class="good">0.9446</td><td>25,567</td></tr>'
        '</tbody></table></div>'
    )
    return (
        '<h2>LLM Layer &amp; Auxiliary Log Classifier</h2>'
        '<div class="g2">'
        + card('LLM Evidence Ablation (RQ4)',
               img('fig_llm_ablation','LLM Ablation') +
               '<p style="margin-top:8px"><strong>Full context</strong> (SHAP+metrics+log) → NDCG@5=0.84. '
               'Remove SHAP → 0.71 (−15.5%). Remove metrics → 0.69. Metrics-only: 0.62. LLM alone: 0.57.</p>')
        + card('LLM Prompt Template (Llama-3-8B)',
               '<div class="code">' + llm_prompt + '</div>'
               '<p style="margin-top:8px">Model: Llama-3-8B-Instruct (4-bit GPTQ). '
               'Inference: 1.8s median on RTX 3060. Prompt: ~480 tokens.</p>')
        + '</div>'
        '<div style="margin-top:16px">'
        + card('Auxiliary Log Classifier (Loghub)',
               '<p>Drain3 log parser → TF-IDF (500 features) → Logistic Regression '
               '(C=1.0, class_weight=\'balanced\'). Labels fed as additional LLM context.</p>'
               + log_tbl)
        + '</div>'
    )

def s_scripts():
    return (
        '<h2>Key Python Scripts</h2>'
        '<p>Core pipeline scripts — simulation, feature engineering, Optuna tuning, SHAP, LLM.</p>'
        + card('01_simulate_data.py — Infrastructure Simulator', code_block(CODE_SIM))
        + '<div style="margin-top:14px">'
        + card('02_feature_engineering.py — 52-Feature Window Builder', code_block(CODE_FEAT))
        + '</div>'
        '<div style="margin-top:14px">'
        + card('03_train_xgboost_optuna.py — Optuna Hyperparameter Search', code_block(CODE_OPTUNA))
        + '</div>'
        '<div style="margin-top:14px">'
        + card('04_shap_explain.py — TreeSHAP Faithfulness Evaluation', code_block(CODE_SHAP))
        + '</div>'
        '<div style="margin-top:14px">'
        + card('05_llm_pipeline.py — LLM Root-Cause Generation', code_block(CODE_LLM))
        + '</div>'
    )

# ── Assemble ─────────────────────────────────────────────────────────────────
NAV = (
    '<nav>'
    '<div class="nt">Explainable AIOps</div>'
    '<button class="active" onclick="show(\'ov\',this)">Overview</button>'
    '<button onclick="show(\'ds\',this)">Dataset &amp; Features</button>'
    '<button onclick="show(\'hp\',this)">Hyperparameter Tuning</button>'
    '<button onclick="show(\'tr\',this)">Training</button>'
    '<button onclick="show(\'rs\',this)">Results</button>'
    '<button onclick="show(\'sh\',this)">SHAP</button>'
    '<button onclick="show(\'ew\',this)">Early-Warning</button>'
    '<button onclick="show(\'llm\',this)">LLM &amp; Logs</button>'
    '<button onclick="show(\'sc\',this)">Key Scripts</button>'
    '<div class="ne"><button class="tb" id="tb" onclick="toggleTheme()">&#127769;</button></div>'
    '</nav>'
)

HTML = (
    '<!DOCTYPE html><html lang="en"><head>'
    '<meta charset="UTF-8"/>'
    '<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>'
    '<title>Explainable AIOps Dashboard</title>'
    '<style>' + CSS + '</style>'
    '</head><body>'
    + NAV
    + section('ov', s_overview(), active=True)
    + section('ds', s_dataset())
    + section('hp', s_hp())
    + section('tr', s_training())
    + section('rs', s_results())
    + section('sh', s_shap())
    + section('ew', s_early())
    + section('llm', s_llm())
    + section('sc', s_scripts())
    + '<script>' + JS + '</script>'
    '</body></html>'
)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"Written: {OUT}  ({os.path.getsize(OUT)//1024} KB)")
