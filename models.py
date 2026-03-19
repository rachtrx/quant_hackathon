import optuna
import numpy as np
from utils import information_coefficient, rank_information_coefficient
from sklearn.metrics import root_mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    pred_valid = model.predict(X_valid)

    rmse = root_mean_squared_error(y_valid, pred_valid)
    ic = information_coefficient(y_valid, pred_valid)
    ric = rank_information_coefficient(y_valid, pred_valid)
    if np.isnan(ric) or np.isinf(ric):
        ric = -1e9

    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("ic", ic)
    trial.set_user_attr("rank_ic", ric)

    trial.report(ric, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return ric

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
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1,
        # optional:
        # "tree_method": "hist",
    }

    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    pred_valid = model.predict(X_valid)

    rmse = root_mean_squared_error(y_valid, pred_valid)
    ic = information_coefficient(y_valid, pred_valid)
    ric = rank_information_coefficient(y_valid, pred_valid)
    if np.isnan(ric) or np.isinf(ric):
        ric = -1e9

    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("ic", ic)
    trial.set_user_attr("rank_ic", ric)

    trial.report(ric, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return ric

MODEL_REGISTRY = {
    "rf": RandomForestRegressor,
    "xgb": XGBRegressor,
}

OBJECTIVES = {
    "rf": _rf_objective,
    "xgb": _xgb_objective
}