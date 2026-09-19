# scripts/prepare_data.py
# PURPOSE: Load from HuggingFace, clean data,
#          split into train/test, upload back to HuggingFace.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os
import json

def load_from_huggingface(hf_username, dataset_repo):
    """Load raw dataset directly from HuggingFace."""
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print("\n── Loading from HuggingFace ──────────────────────")
    url = (
        f"https://huggingface.co/datasets/{hf_username}/"
        f"{dataset_repo}/resolve/main/data/sales_data.csv"
    )
    print(f"  URL: {url}")
    df = pd.read_csv(url)
    print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Remove unnecessary columns and handle missing values."""
    print("\n── Cleaning Data ─────────────────────────────────")

    # Remove ID columns — not useful for prediction
    drop_cols = ["Product_Id", "Store_Id"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  Dropped ID columns: {drop_cols}")

    # Fill missing Product_Weight with median
    if df["Product_Weight"].isnull().sum() > 0:
        med = df["Product_Weight"].median()
        df["Product_Weight"] = df["Product_Weight"].fillna(med)
        print(f"  Product_Weight: filled nulls with median={med:.2f}")

    # Fill missing Store_Size with mode
    if df["Store_Size"].isnull().sum() > 0:
        mode = df["Store_Size"].mode()[0]
        df["Store_Size"] = df["Store_Size"].fillna(mode)
        print(f"  Store_Size: filled nulls with mode='{mode}'")

    # Standardise sugar content values
    df["Product_Sugar_Content"] = df[
        "Product_Sugar_Content"
    ].replace({
        "low fat": "Low Sugar",
        "LF":      "Low Sugar",
        "reg":     "Regular",
    })
    print("  Sugar content values standardised")
    print(f"  Missing after cleaning: {df.isnull().sum().sum()}")
    return df

def add_features(df):
    """Create new features to improve model performance."""
    print("\n── Feature Engineering ───────────────────────────")
    df["Store_Age"] = 2024 - df["Store_Establishment_Year"]
    print("  ✓ Added Store_Age")
    df["Price_Per_Weight"] = (
        df["Product_MRP"] / (df["Product_Weight"] + 0.001)
    )
    print("  ✓ Added Price_Per_Weight")
    return df

def encode_categories(df):
    """Convert text columns to numbers for the model."""
    print("\n── Encoding Categories ───────────────────────────")
    cat_cols     = df.select_dtypes(include="object").columns.tolist()
    encoding_map = {}
    le           = LabelEncoder()
    for col in cat_cols:
        vals          = df[col].unique().tolist()
        df[col]       = le.fit_transform(df[col].astype(str))
        encoding_map[col] = {
            str(v): int(le.transform([str(v)])[0])
            for v in vals
        }
        print(f"  Encoded: {col}")
    os.makedirs("models", exist_ok=True)
    with open("models/encoding_map.json", "w") as f:
        json.dump(encoding_map, f, indent=2)
    print("  ✓ Encoding map saved")
    return df

def split_and_save(df):
    """Split 80/20 train/test and save locally."""
    print("\n── Splitting Dataset ─────────────────────────────")
    X = df.drop("Product_Store_Sales_Total", axis=1)
    y = df["Product_Store_Sales_Total"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_df = X_train.copy()
    train_df["Product_Store_Sales_Total"] = y_train
    test_df  = X_test.copy()
    test_df["Product_Store_Sales_Total"]  = y_test
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",   index=False)
    print(f"  Train: {len(train_df):,} rows → data/train.csv")
    print(f"  Test:  {len(test_df):,} rows  → data/test.csv")

def upload_splits(hf_token, hf_username, dataset_repo):
    """Upload train/test splits back to HuggingFace."""
    print("\n── Uploading Splits to HuggingFace ──────────────")
    api     = HfApi()
    repo_id = f"{hf_username}/{dataset_repo}"
    for local, remote in [
        ("data/train.csv",           "data/train.csv"),
        ("data/test.csv",            "data/test.csv"),
        ("models/encoding_map.json", "models/encoding_map.json"),
    ]:
        if os.path.exists(local):
            api.upload_file(
                path_or_fileobj = local,
                path_in_repo    = remote,
                repo_id         = repo_id,
                repo_type       = "dataset",
                token           = hf_token
            )
            print(f"  ✓ Uploaded: {remote}")
    print(f"  View: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    HF_TOKEN        = os.environ.get("HF_TOKEN", "")
    HF_USERNAME     = os.environ.get("HF_USERNAME", "")
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO",
                                      "sales-forecasting-data")
    df = load_from_huggingface(HF_USERNAME, HF_DATASET_REPO)
    df = clean_data(df)
    df = add_features(df)
    df = encode_categories(df)
    split_and_save(df)
    if HF_TOKEN and HF_USERNAME:
        upload_splits(HF_TOKEN, HF_USERNAME, HF_DATASET_REPO)
    print("\n✓ DATA PREPARATION COMPLETE")
