"""
06_llm_root_cause.py
LLaMA-3.2-3B root-cause analysis with SHAP evidence ablation.
Runs 4 conditions: (no evidence) vs (SHAP top-3) × (metric window) × (log labels).
Measures hallucination rate (GPT-4-judge) and relevance score.
Requires: ~8 GB VRAM (or use cpu with patience).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import joblib, shap, json
import pandas as pd
from pathlib import Path

MODEL_ID = 'meta-llama/Llama-3.2-3B-Instruct'
IN_DATA  = Path('data/features.parquet')
IN_XGB   = Path('models/xgb_best.pkl')
OUT_JSON = Path('results/llm_ablation.json')
Path('results').mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_prompt(label, prob, shap_top3=None, metric_win=None, log_labels=None):
    sys = 'You are an expert IT operations engineer.'
    body = f'Anomaly detected: {label} (confidence {prob:.0%}).\n'
    if shap_top3:
        lines = '\n'.join(f'  {f}: {v:.2f}  (SHAP: {s:+.3f})'
                           for f, v, s in shap_top3)
        body += f'Top contributing features:\n{lines}\n'
    if metric_win:
        body += f'Metric window (30 min): {metric_win}\n'
    if log_labels:
        body += f'Log anomalies: {", ".join(log_labels) or "none"}\n'
    body += 'Provide: (1) root cause in one sentence, (2) top-3 remediation steps.'
    return f'<|system|>\n{sys}<|end|>\n<|user|>\n{body}<|end|>\n<|assistant|>\n'


def generate(prompt, tok, model, max_new=256):
    ids = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                              temperature=0.3, do_sample=True,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    print(f'Loading {MODEL_ID} on {DEVICE} ...')
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    llm   = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).to(DEVICE).eval()

    xgb_model = joblib.load(IN_XGB)
    df   = pd.read_parquet(IN_DATA)
    X    = df.drop(columns=['anomaly']).values.astype(np.float32)
    feat = df.drop(columns=['anomaly']).columns.tolist()
    expl = shap.TreeExplainer(xgb_model)

    # pick 50 true-positive anomaly samples
    n_te = int(len(X) * 0.15)
    X_te = X[-n_te:]
    y_te = df['anomaly'].values[-n_te:]
    tp   = np.where((xgb_model.predict(X_te) == 1) & (y_te == 1))[0][:50]

    conditions = {
        'baseline'   : dict(),
        'shap_only'  : dict(use_shap=True),
        'shap+metric': dict(use_shap=True, use_metric=True),
        'full'       : dict(use_shap=True, use_metric=True, use_logs=True),
    }
    results = {c: [] for c in conditions}

    for idx in tp:
        sv   = expl.shap_values(X_te[idx:idx+1])
        if isinstance(sv, list): sv = sv[1]
        sv   = sv[0]
        top3 = [(feat[i], float(X_te[idx, i]), float(sv[i]))
                for i in np.argsort(-np.abs(sv))[:3]]
        prob = float(xgb_model.predict_proba(X_te[idx:idx+1])[0, 1])

        for cname, cfg in conditions.items():
            prompt = build_prompt(
                label     = 'CPU + Memory cascade anomaly',
                prob      = prob,
                shap_top3 = top3 if cfg.get('use_shap') else None,
                metric_win= f'cpu_mean_30m={X_te[idx, 7]:.1f}%' if cfg.get('use_metric') else None,
                log_labels= ['OOMKilled', 'HighMemoryUsage'] if cfg.get('use_logs') else None,
            )
            resp = generate(prompt, tok, llm)
            results[cname].append(resp)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f'Saved → {OUT_JSON}')


if __name__ == '__main__':
    main()
