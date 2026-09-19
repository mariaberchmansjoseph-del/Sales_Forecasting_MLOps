# app/app.py
# PURPOSE: Gradio web app for sales prediction.
#          Deployed on HuggingFace Spaces with ZeroGPU.
# REQUIRES: import spaces + @spaces.GPU decorator

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import os
import spaces           # Required for ZeroGPU

# ── LOAD MODEL AND SUPPORTING FILES ───────────────────────────
HF_USERNAME   = os.environ.get("HF_USERNAME",   "")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO",
                                "sales-forecasting-model")

def load_model():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id   = f"{HF_USERNAME}/{HF_MODEL_REPO}",
            filename  = "best_model.pkl",
            repo_type = "model"
        )
        return joblib.load(path)
    except Exception as e:
        local = "models/best_model.pkl"
        if os.path.exists(local):
            return joblib.load(local)
        print(f"Model error: {e}")
        return None

def load_meta():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id   = f"{HF_USERNAME}/{HF_MODEL_REPO}",
            filename  = "model_metadata.json",
            repo_type = "model"
        )
        with open(path) as f:
            return json.load(f)
    except:
        local = "models/model_metadata.json"
        if os.path.exists(local):
            with open(local) as f:
                return json.load(f)
        return {}

def load_encoding():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id   = f"{HF_USERNAME}/{HF_MODEL_REPO}",
            filename  = "encoding_map.json",
            repo_type = "model"
        )
        with open(path) as f:
            return json.load(f)
    except:
        local = "models/encoding_map.json"
        if os.path.exists(local):
            with open(local) as f:
                return json.load(f)
        return {}

model        = load_model()
metadata     = load_meta()
encoding_map = load_encoding()

def encode(value, col):
    if col in encoding_map:
        return encoding_map[col].get(str(value), 0)
    return value

def get_model_info():
    if not metadata:
        return "Model info not available."
    m = metadata.get("metrics", {})
    return (
        f"Best Model: {metadata.get('model_name','N/A')}\n"
        f"R² Score:   {m.get('R2',   'N/A')}\n"
        f"RMSE:       {m.get('RMSE', 'N/A')}\n"
        f"MAE:        {m.get('MAE',  'N/A')}"
    )

# ── PREDICTION FUNCTION ───────────────────────────────────────
# @spaces.GPU is REQUIRED for ZeroGPU hardware
@spaces.GPU
def predict_sales(
    product_weight, product_sugar, product_area,
    product_type,   product_mrp,
    store_year,     store_size,
    store_city,     store_type
):
    if model is None:
        return (
            "❌ Model not loaded",
            "N/A", "N/A", "N/A", "N/A",
            "N/A", "Run training pipeline first."
        )

    cat_cols = [
        "Product_Sugar_Content", "Product_Type",
        "Store_Size", "Store_Location_City_Type", "Store_Type"
    ]

    # Collect inputs into a dictionary
    raw = {
        "Product_Weight":           float(product_weight),
        "Product_Sugar_Content":    product_sugar,
        "Product_Allocated_Area":   float(product_area),
        "Product_Type":             product_type,
        "Product_MRP":              float(product_mrp),
        "Store_Establishment_Year": int(store_year),
        "Store_Size":               store_size,
        "Store_Location_City_Type": store_city,
        "Store_Type":               store_type,
        "Store_Age":     2024 - int(store_year),
        "Price_Per_Weight": float(product_mrp) / (
            float(product_weight) + 0.001
        ),
    }

    # Encode categories and save into DataFrame
    encoded  = {
        k: (encode(v, k) if k in cat_cols else v)
        for k, v in raw.items()
    }
    input_df = pd.DataFrame([encoded])
    pred     = float(model.predict(input_df)[0])

    annual    = f"₹{pred:,.2f}"
    monthly   = f"₹{pred/12:,.2f}"
    weekly    = f"₹{pred/52:,.2f}"
    quarterly = f"₹{pred/4:,.2f}"
    daily     = f"₹{pred/365:,.2f}"

    if pred < 1000:
        tier = "🟡 LOW Revenue"
    elif pred < 2500:
        tier = "🔵 MEDIUM Revenue"
    else:
        tier = "🟢 HIGH Revenue"

    insight = (
        f"Annual forecast: {annual}\n"
        f"Weekly target:   {weekly}\n"
        f"Monthly target:  {monthly}\n\n"
        f"Plan procurement based on annual forecast.\n"
        f"Use weekly figure as regional sales benchmark."
    )

    return (annual, monthly, weekly,
            quarterly, daily, tier, insight)

# ── GRADIO UI ─────────────────────────────────────────────────
with gr.Blocks(
    title = "Sales Forecasting MLOps",
    theme = gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 📊 Sales Forecasting — MLOps Pipeline
    Predict product store sales revenue.
    **GitHub Actions CI/CD · HuggingFace Model Hub · ZeroGPU**
    """)

    gr.Textbox(
        label       = "Model Information",
        value       = get_model_info(),
        lines       = 4,
        interactive = False
    )

    gr.Markdown("---")
    gr.Markdown("### Enter Product and Store Details")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Product Details**")
            product_weight = gr.Slider(
                0.1, 50.0, 5.0, step=0.1,
                label="Product Weight (kg)"
            )
            product_sugar = gr.Dropdown(
                ["Low Sugar", "Regular", "No Sugar"],
                value="Regular",
                label="Sugar Content"
            )
            product_area = gr.Slider(
                0.0, 0.3, 0.05, step=0.001,
                label="Product Allocated Area"
            )
            product_type = gr.Dropdown(
                ["Fruits and Vegetables", "Snack Foods",
                 "Household", "Frozen Foods", "Dairy",
                 "Canned", "Baking Goods",
                 "Health and Hygiene", "Soft Drinks",
                 "Meat", "Breads", "Hard Drinks",
                 "Others", "Starchy Foods",
                 "Breakfast", "Seafood"],
                value="Snack Foods",
                label="Product Type"
            )
            product_mrp = gr.Slider(
                10.0, 500.0, 150.0, step=0.5,
                label="Product MRP (₹)"
            )

        with gr.Column():
            gr.Markdown("**Store Details**")
            store_year = gr.Slider(
                1980, 2023, 2005, step=1,
                label="Store Establishment Year"
            )
            store_size = gr.Radio(
                ["Small", "Medium", "High"],
                value="Medium",
                label="Store Size"
            )
            store_city = gr.Radio(
                ["Tier 1", "Tier 2", "Tier 3"],
                value="Tier 2",
                label="City Type"
            )
            store_type = gr.Dropdown(
                ["Supermarket Type1", "Supermarket Type2",
                 "Supermarket Type3", "Grocery Store"],
                value="Supermarket Type1",
                label="Store Type"
            )

    gr.Markdown("---")
    predict_btn = gr.Button(
        "🔮  Predict Sales Revenue",
        variant="primary",
        size="lg"
    )
    gr.Markdown("---")
    gr.Markdown("### Results")

    with gr.Row():
        annual_out    = gr.Textbox(label="Annual Sales",
                                    interactive=False)
        monthly_out   = gr.Textbox(label="Monthly Sales",
                                    interactive=False)
        weekly_out    = gr.Textbox(label="Weekly Sales",
                                    interactive=False)
    with gr.Row():
        quarterly_out = gr.Textbox(label="Quarterly Sales",
                                    interactive=False)
        daily_out     = gr.Textbox(label="Daily Estimate",
                                    interactive=False)
        tier_out      = gr.Textbox(label="Revenue Tier",
                                    interactive=False)

    insight_out = gr.Textbox(
        label="Business Insight",
        lines=5,
        interactive=False
    )

    predict_btn.click(
        fn = predict_sales,
        inputs  = [
            product_weight, product_sugar, product_area,
            product_type,   product_mrp,
            store_year,     store_size,
            store_city,     store_type,
        ],
        outputs = [
            annual_out, monthly_out, weekly_out,
            quarterly_out, daily_out,
            tier_out, insight_out,
        ]
    )

    gr.Examples(
        examples = [
            [12.0, "Low Sugar",  0.08,
             "Snack Foods",      200.0,
             2000, "High",  "Tier 1", "Supermarket Type1"],
            [3.5,  "Regular",   0.03,
             "Fruits and Vegetables", 80.0,
             2015, "Small", "Tier 3", "Grocery Store"],
        ],
        inputs = [
            product_weight, product_sugar, product_area,
            product_type,   product_mrp,
            store_year,     store_size,
            store_city,     store_type,
        ],
        label = "Example inputs — click to auto-fill"
    )

    gr.Markdown(
        "---\n"
        "*Sales Forecasting MLOps · GitHub Actions · "
        "HuggingFace ZeroGPU*"
    )

if __name__ == "__main__":
    demo.launch()
