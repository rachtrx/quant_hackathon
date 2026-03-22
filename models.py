import optuna
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


# =========================================================
# Safe metrics
# =========================================================
def _safe_roc_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _safe_pr_auc(y_true, y_score) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def _safe_log_loss(y_true, y_score) -> float:
    try:
        y_score = np.clip(y_score, 1e-8, 1 - 1e-8)
        return float(log_loss(y_true, y_score))
    except Exception:
        return 1e9


def _safe_brier(y_true, y_score) -> float:
    try:
        return float(brier_score_loss(y_true, y_score))
    except Exception:
        return 1e9


def _compute_metrics(y_true, y_score) -> dict:
    return {
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "log_loss": _safe_log_loss(y_true, y_score),
        "brier": _safe_brier(y_true, y_score),
    }


def _objective_value(metrics: dict, objective_metric: str) -> float:
    if objective_metric == "roc_auc":
        return metrics["roc_auc"]
    if objective_metric == "pr_auc":
        return metrics["pr_auc"]
    if objective_metric == "neg_log_loss":
        return -metrics["log_loss"]
    if objective_metric == "neg_brier":
        return -metrics["brier"]
    raise ValueError(f"Unsupported objective_metric: {objective_metric}")


def _log_trial_metrics(trial: optuna.Trial, metrics: dict) -> None:
    for k, v in metrics.items():
        trial.set_user_attr(k, float(v))


def _get_positive_class_weight(y_train) -> float:
    y_arr = np.asarray(y_train)
    pos = np.sum(y_arr == 1)
    neg = np.sum(y_arr == 0)
    if pos == 0:
        return 1.0
    return max(neg / pos, 1e-6)


# =========================================================
# Optuna objectives
# =========================================================
def _rf_objective(
    trial: optuna.Trial,
    X_train,
    y_train,
    X_valid,
    y_valid,
    objective_metric: str = "roc_auc",
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.3, 0.5, 0.8],
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical(
            "class_weight",
            [None, "balanced", "balanced_subsample"],
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"],
        ),
        "random_state": 42,
        "n_jobs": 64,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    pred_valid = model.predict_proba(X_valid)[:, 1]
    metrics = _compute_metrics(y_valid, pred_valid)
    _log_trial_metrics(trial, metrics)

    score = _objective_value(metrics, objective_metric)
    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


def _xgb_objective(
    trial: optuna.Trial,
    X_train,
    y_train,
    X_valid,
    y_valid,
    objective_metric: str = "roc_auc",
) -> float:
    base_spw = _get_positive_class_weight(y_train)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),

        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),

        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),

        # keep ONLY if imbalance exists
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(0.5, base_spw * 0.8),
            max(1.5, base_spw * 1.2),
            log=True,
        ),

        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 64,
        "tree_method": "hist",
    }

    model = XGBClassifier(
        **params,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    pred_valid = model.predict_proba(X_valid)[:, 1]

    metrics = _compute_metrics(y_valid, pred_valid)
    _log_trial_metrics(trial, metrics)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        trial.set_user_attr("best_iteration", int(best_iteration))

    score = _objective_value(metrics, objective_metric)

    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


OBJECTIVES = {
    "rf": _rf_objective,
    "xgb": _xgb_objective,
}


# =========================================================
# Model builders
# =========================================================
def build_final_rf(best_params: dict) -> RandomForestClassifier:
    params = best_params.copy()
    params["random_state"] = 42
    params["n_jobs"] = -1
    return RandomForestClassifier(**params)


def build_final_xgb(best_params: dict) -> XGBClassifier:
    params = best_params.copy()
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["random_state"] = 42
    params["n_jobs"] = -1
    params["tree_method"] = "hist"
    return XGBClassifier(**params)


FINAL_MODEL_BUILDERS = {
    "rf": build_final_rf,
    "xgb": build_final_xgb,
}


# =========================================================
# Objective factory
# =========================================================
def make_objective(
    model_type: str,
    X_train,
    y_train,
    X_valid,
    y_valid,
    objective_metric: str = "roc_auc",
):
    if model_type not in OBJECTIVES:
        raise ValueError(f"Unsupported model_type: {model_type}")

    def objective(trial: optuna.Trial) -> float:
        return OBJECTIVES[model_type](
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            objective_metric=objective_metric,
        )

    return objective


# =========================================================
# Feature importance helpers
# =========================================================
def get_feature_importance(model, feature_names) -> pd.Series:
    # XGBoost: use gain importance
    if isinstance(model, XGBClassifier):
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        imp = pd.Series(
            [gain_dict.get(f, 0.0) for f in feature_names],
            index=pd.Index(feature_names, name="feature"),
            dtype=float,
        )
        return imp.sort_values(ascending=False)

    # Random Forest: use built-in impurity importance
    if isinstance(model, RandomForestClassifier):
        imp = pd.Series(
            model.feature_importances_,
            index=pd.Index(feature_names, name="feature"),
            dtype=float,
        )
        return imp.sort_values(ascending=False)

    # Fallback for any tree model exposing feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(
            model.feature_importances_,
            index=pd.Index(feature_names, name="feature"),
            dtype=float,
        )
        return imp.sort_values(ascending=False)

    raise ValueError("Model does not expose usable feature importance.")


def select_top_features(
    importance: pd.Series,
    top_k: int,
    min_features: int = 10,
) -> list[str]:
    """
    Pick either:
    - top_k features
    """
    if top_k is None:
        raise ValueError("top_k value must be provided")

    top_k = min(top_k, len(importance))
    top_k = max(top_k, min_features)
    return importance.head(top_k).index.tolist()

def select_features_only(
    model_type: str,
    X_train: pd.DataFrame,
    y_train,
    top_k: int | None = None,
    min_features: int = 10,
):
    """
    Fit a cheap selector model on all features only to rank features.
    This is NOT the final tuned model.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame with column names.")

    if model_type == "xgb":
        selector_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-3,
            reg_lambda=1.0,
            gamma=1e-6,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    elif model_type == "rf":
        selector_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    selector_model.fit(X_train, y_train)

    importance = get_feature_importance(selector_model, X_train.columns)
    selected_features = select_top_features(
        importance=importance,
        top_k=top_k,
        min_features=min_features,
    )

    return {
        "selector_model": selector_model,
        "feature_importance": importance,
        "selected_features": selected_features,
    }


# =========================================================
# Single-stage tuning on already-selected features
# =========================================================
def tune_model(
    model_type: str,
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials: int = 100,
    objective_metric: str = "roc_auc",
    study_name: str | None = None,
    sampler_seed: int = 42,
):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )

    objective = make_objective(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        objective_metric=objective_metric,
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params.copy()
    model = FINAL_MODEL_BUILDERS[model_type](best_params)

    if model_type == "xgb":
        model.set_params(early_stopping_rounds=100)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    pred_valid = model.predict_proba(X_valid)[:, 1]
    metrics = _compute_metrics(y_valid, pred_valid)

    return {
        "study": study,
        "best_params": best_params,
        "model": model,
        "pred_valid": pred_valid,
        "metrics": metrics,
    }


# =========================================================
# Select features first, then tune only selected features
# =========================================================
def tune_selected_features_only(
    model_type: str,
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    n_trials: int = 100,
    objective_metric: str = "roc_auc",
    top_k: int | None = None,
    min_features: int = 10,
    study_name: str | None = None,
):
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_valid, pd.DataFrame):
        raise ValueError("X_train and X_valid should be pandas DataFrames with column names.")

    fs_res = select_features_only(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        top_k=top_k,
        min_features=min_features,
    )

    selected_features = fs_res["selected_features"]

    X_train_sel = X_train[selected_features].copy()
    X_valid_sel = X_valid[selected_features].copy()

    tune_res = tune_model(
        model_type=model_type,
        X_train=X_train_sel,
        y_train=y_train,
        X_valid=X_valid_sel,
        y_valid=y_valid,
        n_trials=n_trials,
        objective_metric=objective_metric,
        study_name=study_name,
    )

    return {
        "model_type": model_type,
        "objective_metric": objective_metric,
        "feature_importance": fs_res["feature_importance"],
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "X_train_selected": X_train_sel,
        "X_valid_selected": X_valid_sel,
        "study": tune_res["study"],
        "best_params": tune_res["best_params"],
        "model": tune_res["model"],
        "pred_valid": tune_res["pred_valid"],
        "metrics": tune_res["metrics"],
    }


# =========================================================
# Optional forward-return bucket diagnostic
# =========================================================
def make_bucket_table(pred, y_true, fwd_ret, n_bins: int = 10) -> pd.DataFrame:
    eval_df = pd.DataFrame({
        "pred": np.asarray(pred),
        "y_true": np.asarray(y_true),
        "fwd_ret": np.asarray(fwd_ret),
    }).dropna()

    eval_df["pred_bin"] = pd.qcut(eval_df["pred"], n_bins, duplicates="drop")

    bucket_stats = eval_df.groupby("pred_bin", observed=False).agg(
        mean_pred=("pred", "mean"),
        pos_rate=("y_true", "mean"),
        mean_fwd_ret=("fwd_ret", "mean"),
        std_fwd_ret=("fwd_ret", "std"),
        count=("fwd_ret", "count"),
    )

    return bucket_stats


# =========================================================
# Final refit on full train+valid data, but only selected features
# =========================================================
def fit_final_model(
    model_type: str,
    best_params: dict,
    selected_features: list[str],
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
):
    X_train_sel = X_train[selected_features].copy()
    X_valid_sel = X_valid[selected_features].copy()

    base_model = FINAL_MODEL_BUILDERS[model_type](best_params)
    base_model.fit(X_train_sel, y_train)

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
    )
    calibrator.fit(X_valid_sel, y_valid)

    return {
        "base_model": base_model,
        "calibrator": calibrator,
        "selected_features": selected_features,
    }