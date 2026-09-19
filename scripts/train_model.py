import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor
)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from huggingface_hub import HfApi

def load_data(hf_username, dataset_repo):
    """Load train and test data from HuggingFace."""
    print("=" * 60)
    print("MODEL TRAINING AND REGISTRATION")
    print("=" * 60)
    print("\n── Loading Data from HuggingFace ─────────────────")
    base = (
        f"https://huggingface.co/datasets/{hf_username}/"
        f"{dataset_repo}/resolve/main"
    )
    train_df = pd.read_csv(f"{base}/data/train.csv")
    test_df  = pd.read_csv(f"{base}/data/test.csv")
    print(f"  ✓ Train: {train_df.shape[0]:,} rows")
    print(f"  ✓ Test:  {test_df.shape[0]:,} rows")
    X_train = train_df.drop("Product_Store_Sales_Total", axis=1)
    y_train = train_df["Product_Store_Sales_Total"]
    X_test  = test_df.drop("Product_Store_Sales_Total",  axis=1)
    y_test  = test_df["Product_Store_Sales_Total"]
    return X_train, X_test, y_train, y_test

def get_models():
    """Define 6 models with hyperparameter grids."""
    return {
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth":         [5, 10, 15],
                "min_samples_split": [2, 5],
                "criterion":         ["squared_error",
                                      "absolute_error"]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                random_state=42, n_jobs=-1
            ),
            "params": {
                "n_estimators": [50, 100],
                "max_depth":    [5, 10],
                "max_features": ["sqrt", "log2"]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(
                random_state=42
            ),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5]
            }
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.5, 1.0]
            }
        },
        "XGBoost": {
            "model": XGBRegressor(
                random_state=42,
                eval_metric="rmse",
                verbosity=0
            ),
            "params": {
                "n_estimators":  [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5]
            }
        },
        "Bagging": {
            "model": BaggingRegressor(
                random_state=42, n_jobs=-1
            ),
            "params": {
                "n_estimators": [10, 20],
                "max_samples":  [0.8, 1.0]
            }
        }
    }

def evaluate(model, X_test, y_test):
    """Calculate MAE, RMSE, R2 metrics."""
    y_pred = model.predict(X_test)
    return {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(
            mean_squared_error(y_test, y_pred)), 2),
        "R2":   round(r2_score(y_test, y_pred), 4)
    }

def train_all(X_train, X_test, y_train, y_test):
    """Train all 6 models and track with MLflow."""
    print("\n── Training 6 Models ─────────────────────────────")

    # Set MLflow tracking URI BEFORE any logging
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sales-forecasting")

    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    models     = get_models()
    results    = {}
    best_score = -999
    best_model = None
    best_name  = None

    for name, config in models.items():
        print(f"\n  Training {name}...")

        with mlflow.start_run(run_name=name):

            # Hyperparameter tuning
            gs = GridSearchCV(
                estimator  = config["model"],
                param_grid = config["params"],
                cv         = cv,
                scoring    = "r2",
                n_jobs     = -1
            )
            gs.fit(X_train, y_train)

            best_est = gs.best_estimator_
            cv_r2    = gs.best_score_
            metrics  = evaluate(best_est, X_test, y_test)

            # Log parameters to MLflow
            mlflow.log_param("model_name", name)
            for k, v in gs.best_params_.items():
                mlflow.log_param(k, v)

            # Log metrics to MLflow
            mlflow.log_metric("cv_r2",     round(cv_r2, 4))
            mlflow.log_metric("test_r2",   metrics["R2"])
            mlflow.log_metric("test_rmse", metrics["RMSE"])
            mlflow.log_metric("test_mae",  metrics["MAE"])

            # ── FIX: use cloudpickle to avoid skops error ──────
            # serialization_format="cloudpickle" bypasses the
            # UntrustedTypesFoundException for all sklearn types
            mlflow.sklearn.log_model(
                sk_model             = best_est,
                artifact_path        = "model",
                serialization_format = "cloudpickle"
            )

            print(f"    Best params: {gs.best_params_}")
            print(f"    CV R2:       {cv_r2:.4f}")
            print(f"    Test R2:     {metrics['R2']:.4f}")
            print(f"    Test RMSE:   {metrics['RMSE']:.2f}")
            print(f"    Test MAE:    {metrics['MAE']:.2f}")

            results[name] = {
                "model":   best_est,
                "metrics": metrics
            }

            # Track best model
            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_model = best_est
                best_name  = name

    return results, best_model, best_name, best_score

def print_comparison(results):
    """Print model comparison table."""
    print("\n── Model Comparison ──────────────────────────────")
    print(f"  {'Model':20} {'R2':8} {'RMSE':10} {'MAE':10}")
    print("  " + "-" * 50)
    sorted_r = sorted(
        results.items(),
        key    = lambda x: x[1]["metrics"]["R2"],
        reverse = True
    )
    for name, r in sorted_r:
        m = r["metrics"]
        print(
            f"  {name:20} {m['R2']:8.4f} "
            f"{m['RMSE']:10.2f} {m['MAE']:10.2f}"
        )

def save_and_register(model, name, metrics,
                       hf_token, hf_username, model_repo):
    """Save model locally and upload to HuggingFace Model Hub."""
    print(f"\n── Saving Best Model ({name}) ────────────────────")

    os.makedirs("models", exist_ok=True)

    # Save with joblib (reliable, no skops issues)
    joblib.dump(model, "models/best_model.pkl")
    print("  ✓ Saved: models/best_model.pkl")

    # Save metadata
    metadata = {
        "model_name": name,
        "metrics":    metrics,
        "task":       "regression",
        "target":     "Product_Store_Sales_Total"
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved: models/model_metadata.json")

    if not (hf_token and hf_username):
        print("  WARNING: HF credentials missing. Skipping upload.")
        return

    # Upload to HuggingFace Model Hub
    print("\n── Registering to HuggingFace Model Hub ─────────")
    api     = HfApi()
    repo_id = f"{hf_username}/{model_repo}"

    model_card = f"""---
language: en
license: mit
tags:
  - sales-forecasting
  - regression
---
# Sales Forecasting Model
**Algorithm:** {name}
**R²:**   {metrics['R2']}
**RMSE:** {metrics['RMSE']}
**MAE:**  {metrics['MAE']}
"""
    with open("models/README.md", "w") as f:
        f.write(model_card)

    for local, remote in [
        ("models/best_model.pkl",      "best_model.pkl"),
        ("models/model_metadata.json", "model_metadata.json"),
        ("models/encoding_map.json",   "encoding_map.json"),
        ("models/README.md",           "README.md"),
    ]:
        if os.path.exists(local):
            api.upload_file(
                path_or_fileobj = local,
                path_in_repo    = remote,
                repo_id         = repo_id,
                repo_type       = "model",
                token           = hf_token
            )
            print(f"  ✓ Uploaded: {remote}")
        else:
            print(f"  ⚠ Missing: {local}")

    print(f"  View: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    HF_TOKEN        = os.environ.get("HF_TOKEN", "")
    HF_USERNAME     = os.environ.get("HF_USERNAME", "")
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO",
                                      "sales-forecasting-data")
    HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO",
                                      "sales-forecasting-model")

    X_train, X_test, y_train, y_test = load_data(
        HF_USERNAME, HF_DATASET_REPO
    )
    results, best_model, best_name, best_score = train_all(
        X_train, X_test, y_train, y_test
    )
    print_comparison(results)
    save_and_register(
        best_model,
        best_name,
        results[best_name]["metrics"],
        HF_TOKEN,
        HF_USERNAME,
        HF_MODEL_REPO
    )
    print(f"\n{'='*60}")
    print(f"✓ TRAINING COMPLETE")
    print(f"  Best Model: {best_name}")
    print(f"  Best R2:    {best_score:.4f}")
    print("=" * 60)
