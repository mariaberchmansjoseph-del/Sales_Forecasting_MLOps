# scripts/register_data.py
# PURPOSE: Validate the sales dataset and upload to
#          HuggingFace Dataset Hub.

import pandas as pd
import os
import sys
from huggingface_hub import HfApi

def validate_dataset(filepath):
    """Check dataset has correct columns and print summary."""
    print("=" * 60)
    print("DATA REGISTRATION AND VALIDATION")
    print("=" * 60)

    # Check file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    print(f"✓ File found: {filepath}")

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Check expected columns
    expected = [
        "Product_Id", "Product_Weight",
        "Product_Sugar_Content", "Product_Allocated_Area",
        "Product_Type", "Product_MRP", "Store_Id",
        "Store_Establishment_Year", "Store_Size",
        "Store_Location_City_Type", "Store_Type",
        "Product_Store_Sales_Total"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
    else:
        print(f"✓ All {len(expected)} columns present")

    # Print summary
    print(f"\n── Summary ───────────────────────────────────────")
    print(f"  Rows:           {df.shape[0]:,}")
    print(f"  Columns:        {df.shape[1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates:     {df.duplicated().sum()}")
    print(f"\n── Target Variable (Sales) ───────────────────────")
    print(f"  Min:    {df['Product_Store_Sales_Total'].min():.2f}")
    print(f"  Max:    {df['Product_Store_Sales_Total'].max():.2f}")
    print(f"  Mean:   {df['Product_Store_Sales_Total'].mean():.2f}")
    print(f"  Median: {df['Product_Store_Sales_Total'].median():.2f}")
    print(f"\n── Missing Per Column ────────────────────────────")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if len(mv) == 0:
        print("  No missing values.")
    else:
        for col, cnt in mv.items():
            print(f"  {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    print("\n" + "=" * 60)
    print("✓ VALIDATION COMPLETE")
    print("=" * 60)
    return df

def upload_to_huggingface(filepath, hf_token,
                           hf_username, dataset_repo):
    """Upload raw dataset to HuggingFace Dataset Hub."""
    print("\n── Uploading to HuggingFace ──────────────────────")
    api     = HfApi()
    repo_id = f"{hf_username}/{dataset_repo}"
    api.upload_file(
        path_or_fileobj = filepath,
        path_in_repo    = "data/sales_data.csv",
        repo_id         = repo_id,
        repo_type       = "dataset",
        token           = hf_token
    )
    print(f"✓ Uploaded to: {repo_id}")
    print(f"  View: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    HF_TOKEN        = os.environ.get("HF_TOKEN", "")
    HF_USERNAME     = os.environ.get("HF_USERNAME", "")
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO",
                                      "sales-forecasting-data")
    df = validate_dataset("data/sales_data.csv")
    if HF_TOKEN and HF_USERNAME:
        upload_to_huggingface(
            "data/sales_data.csv",
            HF_TOKEN, HF_USERNAME, HF_DATASET_REPO
        )
    else:
        print("WARNING: HF credentials not set. Skipping upload.")
