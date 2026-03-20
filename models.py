import optuna
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


def _safe_roc_auc(y_true, y_score) -> float:
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5


def _safe_pr_auc(y_true, y_score) -> float:
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return 0.0


def _safe_log_loss(y_true, y_score) -> float:
    try:
        y_score = np.clip(y_score, 1e-8, 1 - 1e-8)
        return log_loss(y_true, y_score)
    except Exception:
        return 1e9


def _safe_brier(y_true, y_score) -> float:
    try:
        return brier_score_loss(y_true, y_score)
    except Exception:
        return 1e9


def _rf_objective(trial: optuna.Trial, X_train, y_train, X_valid, y_valid) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
        "random_state": 42,
        "n_jobs": 64,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    pred_valid = model.predict_proba(X_valid)[:, 1]

    roc_auc = _safe_roc_auc(y_valid, pred_valid)
    pr_auc = _safe_pr_auc(y_valid, pred_valid)
    ll = _safe_log_loss(y_valid, pred_valid)
    brier = _safe_brier(y_valid, pred_valid)

    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("pr_auc", pr_auc)
    trial.set_user_attr("log_loss", ll)
    trial.set_user_attr("brier", brier)

    trial.report(roc_auc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return roc_auc


def _xgb_objective(trial: optuna.Trial, X_train, y_train, X_valid, y_valid) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 64,
        # "tree_method": "hist",
    }

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    pred_valid = model.predict_proba(X_valid)[:, 1]

    roc_auc = _safe_roc_auc(y_valid, pred_valid)
    pr_auc = _safe_pr_auc(y_valid, pred_valid)
    ll = _safe_log_loss(y_valid, pred_valid)
    brier = _safe_brier(y_valid, pred_valid)

    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("pr_auc", pr_auc)
    trial.set_user_attr("log_loss", ll)
    trial.set_user_attr("brier", brier)

    trial.report(roc_auc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return roc_auc


MODEL_REGISTRY = {
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}

OBJECTIVES = {
    "rf": _rf_objective,
    "xgb": _xgb_objective,
}