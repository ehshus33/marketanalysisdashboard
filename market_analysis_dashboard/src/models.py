from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix

from .features import FEATURE_COLUMNS



def split_time_series(data: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    return train, test

def train_direction_model(train: pd.DataFrame, model_name: str):
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_up"]

    if model_name == "Logistic Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ])
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])

    model.fit(X_train, y_train)
    return model

def evaluate_direction_model(model, test: pd.DataFrame):
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target_up"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X_test), np.nan)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, y_pred, y_prob, cm

def train_return_model(train: pd.DataFrame, model_name: str):
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_return"]

    if model_name == "Linear Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])

    model.fit(X_train, y_train)
    return model

def evaluate_return_model(model, test: pd.DataFrame):
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target_return"]
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    return metrics, y_pred

def latest_prediction(direction_model, return_model, full_data: pd.DataFrame):
    latest = full_data.iloc[[-1]]
    X_latest = latest[FEATURE_COLUMNS]

    prob_up = float(direction_model.predict_proba(X_latest)[:, 1][0]) if hasattr(direction_model, "predict_proba") else float("nan")
    class_pred = int(direction_model.predict(X_latest)[0])
    predicted_return = float(return_model.predict(X_latest)[0])

    return {
        "direction": "Up" if class_pred == 1 else "Down",
        "prob_up": prob_up,
        "predicted_return": predicted_return,
    }

def baseline_accuracy(test: pd.DataFrame) -> float:
    baseline_pred = (test["return_1d"] > 0).astype(int)
    actual = test["target_up"]
    baseline_pred = baseline_pred.iloc[:-1]
    actual = actual.iloc[:-1]
    accuracy = (baseline_pred == actual).mean()
    return float(accuracy)