"""
03_train_tabular.py
Train XGBoost, Random Forest, and SVM classifiers on the 52-feature matrix.
Hyperparameter optimisation via Optuna (50 trials per model).
Reports held-out Precision / Recall / F1 / AUC.
Input:  data/features.parquet
Output: models/xgb_best.pkl  models/rf_best.pkl  models/svm_best.pkl
        figures/ (Optuna visualisations)
"""
import numpy as np
import pandas as pd
import joblib, optuna
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

IN = Path('data/features.parquet')
Path('models').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

SEED = 42


def load_data():
    df = pd.read_parquet(IN)
    X  = df.drop(columns=['anomaly']).values.astype(np.float32)
    y  = df['anomaly'].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, shuffle=False)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15, shuffle=False)
    return X_tr, X_va, X_te, y_tr, y_va, y_te, df.drop(columns=['anomaly']).columns.tolist()


def score(name, model, X_te, y_te):
    yp = model.predict(X_te)
    pr = precision_score(y_te, yp, zero_division=0)
    rc = recall_score(y_te, yp)
    f1 = f1_score(y_te, yp)
    try:
        pp = model.predict_proba(X_te)[:, 1]
        au = roc_auc_score(y_te, pp)
    except Exception:
        au = float('nan')
    print(f'{name:20s}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}  AUC={au:.3f}')
    return f1


# ── XGBoost ──────────────────────────────────────────────────────────────────
def tune_xgb(X_tr, X_va, y_tr, y_va):
    def obj(trial):
        p = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 1000),
            max_depth        = trial.suggest_int('max_depth', 3, 10),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        )
        m = xgb.XGBClassifier(**p, tree_method='hist', eval_metric='logloss',
                               use_label_encoder=False, random_state=SEED)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              early_stopping_rounds=20, verbose=False)
        return f1_score(y_va, m.predict(X_va))

    study = optuna.create_study(direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=50, show_progress_bar=True)
    print(f'XGB best val F1={study.best_value:.4f}  params={study.best_params}')

    # Save Optuna visualisation
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_param_importances(study, ax=ax)
        fig.tight_layout()
        fig.savefig('figures/fig_optuna_xgb.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return study.best_params


def build_xgb(params, X_tr, y_tr):
    m = xgb.XGBClassifier(**params, tree_method='hist',
                           eval_metric='logloss',
                           use_label_encoder=False, random_state=SEED)
    m.fit(X_tr, y_tr)
    return m


# ── Random Forest ─────────────────────────────────────────────────────────────
def tune_rf(X_tr, X_va, y_tr, y_va):
    def obj(trial):
        p = dict(
            n_estimators    = trial.suggest_int('n_estimators', 50, 500),
            max_depth       = trial.suggest_int('max_depth', 3, 20),
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
            max_features    = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            class_weight    = trial.suggest_categorical('class_weight',
                                ['balanced', 'balanced_subsample', None]),
        )
        m = RandomForestClassifier(**p, n_jobs=-1, random_state=SEED)
        m.fit(X_tr, y_tr)
        return f1_score(y_va, m.predict(X_va))

    study = optuna.create_study(direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=50, show_progress_bar=True)
    print(f'RF  best val F1={study.best_value:.4f}  params={study.best_params}')

    try:
        from optuna.visualization.matplotlib import plot_param_importances
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_param_importances(study, ax=ax)
        fig.tight_layout()
        fig.savefig('figures/fig_optuna_rf.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return study.best_params


# ── SVM ───────────────────────────────────────────────────────────────────────
def tune_svm(X_tr, X_va, y_tr, y_va):
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_va = sc.transform(X_tr), sc.transform(X_va)

    def obj(trial):
        C     = trial.suggest_float('C', 0.01, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        m = SVC(C=C, gamma=gamma, probability=True, random_state=SEED)
        m.fit(Xs_tr, y_tr)
        return f1_score(y_va, m.predict(Xs_va))

    study = optuna.create_study(direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=30, show_progress_bar=True)
    print(f'SVM best val F1={study.best_value:.4f}  params={study.best_params}')
    return study.best_params, sc


if __name__ == '__main__':
    X_tr, X_va, X_te, y_tr, y_va, y_te, feat_names = load_data()
    print(f'Train {X_tr.shape} | Val {X_va.shape} | Test {X_te.shape}')
    print(f'Anomaly rate — train {y_tr.mean():.2%} | test {y_te.mean():.2%}\n')

    print('=== XGBoost ===')
    xgb_params = tune_xgb(X_tr, X_va, y_tr, y_va)
    xgb_model  = build_xgb(xgb_params, np.vstack([X_tr, X_va]),
                            np.hstack([y_tr, y_va]))
    joblib.dump(xgb_model, 'models/xgb_best.pkl')

    print('\n=== Random Forest ===')
    rf_params = tune_rf(X_tr, X_va, y_tr, y_va)
    rf_model  = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=SEED)
    rf_model.fit(np.vstack([X_tr, X_va]), np.hstack([y_tr, y_va]))
    joblib.dump(rf_model, 'models/rf_best.pkl')

    print('\n=== SVM ===')
    svm_params, scaler = tune_svm(X_tr, X_va, y_tr, y_va)
    svm_model = SVC(**svm_params, probability=True, random_state=SEED)
    svm_model.fit(scaler.transform(np.vstack([X_tr, X_va])),
                  np.hstack([y_tr, y_va]))
    joblib.dump((svm_model, scaler), 'models/svm_best.pkl')

    print('\n=== Held-Out Results ===')
    score('XGBoost',       xgb_model, X_te, y_te)
    score('Random Forest', rf_model,  X_te, y_te)
    score('SVM',           svm_model, scaler.transform(X_te), y_te)
