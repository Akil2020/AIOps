"""
04_train_sequence.py
Train LSTM, GRU, and PatchTST sequence models on 30-step windows.
Input:  data/features.parquet
Output: models/lstm_best.pt  models/gru_best.pt  models/patchtst_best.pt
        figures/fig_loss_*.png
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN = Path('data/features.parquet')
Path('models').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

SEQ_LEN   = 30
BATCH     = 256
EPOCHS    = 40
LR        = 1e-3
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def load():
    df   = pd.read_parquet(IN)
    X    = df.drop(columns=['anomaly']).values.astype(np.float32)
    y    = df['anomaly'].values.astype(int)
    n    = len(X)
    tr_e = int(n * 0.70)
    va_e = int(n * 0.85)
    Xtr, ytr = make_sequences(X[:tr_e],  y[:tr_e])
    Xva, yva = make_sequences(X[tr_e:va_e], y[tr_e:va_e])
    Xte, yte = make_sequences(X[va_e:],  y[va_e:])
    to_dl = lambda Xs, ys, sh=True: DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
        batch_size=BATCH, shuffle=sh)
    return to_dl(Xtr, ytr), to_dl(Xva, yva, False),            to_dl(Xte, yte, False), X.shape[1]


# ── Models ────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, inp, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.rnn = nn.LSTM(inp, hidden, layers, batch_first=True,
                           dropout=drop, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(1)


class GRUModel(nn.Module):
    def __init__(self, inp, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.rnn = nn.GRU(inp, hidden, layers, batch_first=True,
                          dropout=drop, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(1)


class PatchTST(nn.Module):
    """Simplified PatchTST: patch the time series, then Transformer encoder."""
    def __init__(self, inp, seq_len=SEQ_LEN, patch_len=5, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.patch_len  = patch_len
        n_patches = seq_len // patch_len
        self.proj  = nn.Linear(inp * patch_len, d_model)
        enc_layer  = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                                dropout=0.1, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, layers)
        self.head  = nn.Linear(d_model, 1)

    def forward(self, x):                         # x: (B, T, F)
        B, T, F = x.shape
        P = self.patch_len
        x = x[:, :T - T % P, :]                  # trim to multiple of P
        x = x.reshape(B, -1, P * F)              # (B, n_patches, P*F)
        x = self.proj(x)                          # (B, n_patches, d_model)
        x = self.enc(x)
        return self.head(x[:, -1, :]).squeeze(1)


def train_model(model, tr_dl, va_dl, name):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0]).to(DEVICE))

    tr_losses, va_losses = [], []
    best_f1, best_state  = 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = 0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        tr_losses.append(tot / len(tr_dl))

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            vl = 0
            for Xb, yb in va_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logit = model(Xb)
                vl += crit(logit, yb).item()
                preds.extend((torch.sigmoid(logit) > 0.5).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        va_losses.append(vl / len(va_dl))
        f1 = f1_score(trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'  [{name}] epoch {epoch:3d}  tr={tr_losses[-1]:.4f}  va={va_losses[-1]:.4f}  F1={f1:.3f}')

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(tr_losses, label='train')
    ax.plot(va_losses, label='val')
    ax.set_title(f'{name} Loss')
    ax.legend(); ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    fig.tight_layout()
    fig.savefig(f'figures/fig_loss_{name.lower()}.png', dpi=150)
    plt.close(fig)

    model.load_state_dict(best_state)
    return model, best_f1


if __name__ == '__main__':
    tr_dl, va_dl, te_dl, n_feat = load()
    print(f'Feature dim={n_feat}  device={DEVICE}')

    results = {}
    for ModelClass, mname in [
        (LSTMModel, 'LSTM'), (GRUModel, 'GRU'), (PatchTST, 'PatchTST')
    ]:
        print(f'\n=== {mname} ===')
        model, val_f1 = train_model(ModelClass(n_feat), tr_dl, va_dl, mname)
        torch.save(model.state_dict(), f'models/{mname.lower()}_best.pt')

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xb, yb in te_dl:
                logit = model(Xb.to(DEVICE))
                preds.extend((torch.sigmoid(logit) > 0.5).cpu().numpy())
                trues.extend(yb.numpy())
        f1 = f1_score(trues, preds, zero_division=0)
        print(f'  {mname} test F1 = {f1:.3f}')
        results[mname] = f1

    print('\n=== Sequence Model Summary ===')
    for k, v in results.items():
        print(f'  {k:12s} F1={v:.3f}')
