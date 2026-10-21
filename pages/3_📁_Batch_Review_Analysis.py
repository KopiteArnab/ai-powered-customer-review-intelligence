"""
ReviewIQ — Page 3: Batch Review Analysis
Upload, validate, and process batch customer review CSV files.
Provides filtering, analytics visualizations, and export capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from src.ui_helpers import render_sidebar, apply_plotly_theme
from src.validation import validate_uploaded_csv, detect_candidate_columns, validate_rating_values
from src.prediction import predict_batch

st.set_page_config(
    page_title="Batch Review Analysis | ReviewIQ",
    page_icon="📁",
    layout="wide"
)

artifacts = render_sidebar("Batch Analysis")

st.title("📁 Batch Review Analysis")
st.markdown(
    "Upload a CSV file containing customer reviews to generate batch rating predictions, "
    "negative sentiment probabilities, and priority triage flags."
)

# Upload Section
uploaded_file = st.file_uploader(
    "Upload Customer Reviews CSV (Max 50MB)",
    type=["csv"],
    help="Upload standard comma-separated customer reviews dataset."
)

col_sample, col_info = st.columns([1, 2])
with col_sample:
    if st.button("📥 Load Built-in Sample CSV (20 Reviews)"):
        with open("data/sample_reviews.csv", "rb") as f:
            sample_bytes = f.read()
        is_valid, msg, raw_df = validate_uploaded_csv(sample_bytes, "sample_reviews.csv")
        st.session_state["uploaded_raw_df"] = raw_df
        st.session_state["uploaded_filename"] = "sample_reviews.csv"
        st.rerun()

# Check uploaded file
raw_df = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    is_valid, msg, df_parsed = validate_uploaded_csv(file_bytes, uploaded_file.name)
    if not is_valid:
        st.error(f"❌ Validation Error: {msg}")
        st.stop()
    else:
        st.success(f"✅ {msg}")
        raw_df = df_parsed
        st.session_state["uploaded_raw_df"] = raw_df
        st.session_state["uploaded_filename"] = uploaded_file.name
elif "uploaded_raw_df" in st.session_state:
    raw_df = st.session_state["uploaded_raw_df"]

if raw_df is None:
    st.info("👆 Please upload a customer reviews CSV or click 'Load Built-in Sample CSV' above to begin.")
    st.stop()

# Preview & Column Configuration
st.subheader("1. Inspect Schema & Map Columns")
st.dataframe(raw_df.head(4), use_container_width=True)

detected = detect_candidate_columns(raw_df)
cols = list(raw_df.columns)

c1, c2, c3, c4 = st.columns(4)
with c1:
    text_default_idx = cols.index(detected["text_col"]) if detected["text_col"] in cols else 0
    selected_text_col = st.selectbox("Review Text Column (Required)", options=cols, index=text_default_idx)
with c2:
    title_options = ["(None / No Title)"] + cols
    title_default_idx = (cols.index(detected["title_col"]) + 1) if detected["title_col"] in cols else 0
    selected_title_col = st.selectbox("Review Title Column (Optional)", options=title_options, index=title_default_idx)
with c3:
    rating_options = ["(None / No Ground Truth)"] + cols
    rating_default_idx = (cols.index(detected["rating_col"]) + 1) if detected["rating_col"] in cols else 0
    selected_rating_col = st.selectbox("Actual Rating Column (Optional)", options=rating_options, index=rating_default_idx)
with c4:
    prod_options = ["(None)"] + cols
    prod_default_idx = (cols.index(detected["product_col"]) + 1) if detected["product_col"] in cols else 0
    selected_prod_col = st.selectbox("Product / Brand Column (Optional)", options=prod_options, index=prod_default_idx)

# Date column detection
date_options = ["(None)"] + cols
date_default_idx = (cols.index(detected["date_col"]) + 1) if detected["date_col"] in cols else 0
selected_date_col = st.selectbox("Date Column (Optional)", options=date_options, index=date_default_idx)

# Validation diagnostics
if selected_rating_col != "(None / No Ground Truth)":
    rating_val = validate_rating_values(raw_df[selected_rating_col])
    if not rating_val["is_usable_for_eval"]:
        st.warning(
            f"Notice: '{selected_rating_col}' contains only {rating_val['valid_count']} valid 1-5 integer ratings. "
            "Inference will proceed, but ground-truth evaluations may be restricted."
        )

st.markdown("---")

# Batch Prediction Action
st.subheader("2. Run Batch Inference")
col_run, col_status = st.columns([1, 3])
with col_run:
    run_batch = st.button("🚀 Process Reviews Pipeline", type="primary", use_container_width=True)

# Run or use cached predictions
if run_batch or ("batch_results_df" in st.session_state and st.session_state.get("batch_source_file") == st.session_state.get("uploaded_filename")):
    if run_batch:
        with st.spinner(f"Executing NLP inference on {len(raw_df):,} reviews..."):
            actual_title_col = selected_title_col if selected_title_col != "(None / No Title)" else None
            results_df = predict_batch(
                raw_df,
                text_col=selected_text_col,
                title_col=actual_title_col,
                artifacts=artifacts
            )
            st.session_state["batch_results_df"] = results_df
            st.session_state["batch_source_file"] = st.session_state.get("uploaded_filename")
            # Update global session state for dashboard
            st.session_state["active_data"] = results_df
            st.session_state["text_col"] = selected_text_col
            st.session_state["title_col"] = actual_title_col
            st.session_state["rating_col"] = selected_rating_col if selected_rating_col != "(None / No Ground Truth)" else None
            st.session_state["product_col"] = selected_prod_col if selected_prod_col != "(None)" else None
            st.session_state["date_col"] = selected_date_col if selected_date_col != "(None)" else None

    results_df = st.session_state.get("batch_results_df", raw_df)

    st.success(f"✅ Successfully processed {len(results_df):,} customer reviews!", icon="🎯")

    # Summary Metrics Row
    total_batch = len(results_df)
    avg_pred = float(results_df["predicted_rating"].mean())
    neg_count = int(results_df["is_negative_predicted"].sum())
    neg_pct = round(neg_count / total_batch * 100, 1) if total_batch > 0 else 0.0
    high_p_count = int((results_df["priority_level"] == "High Priority").sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Reviews Processed", f"{total_batch:,}")
    with m2:
        st.metric("Predicted Average Rating", f"{avg_pred:.2f} ★")
    with m3:
        st.metric("Negative Review Share", f"{neg_pct}%", f"{neg_count} reviews", delta_color="inverse")
    with m4:
        st.metric("High-Priority Support Queue", f"{high_p_count:,}")

    st.markdown("---")

    # Filters Section
    st.subheader("3. Explore & Filter Batch Predictions")
    f1, f2, f3 = st.columns(3)
    with f1:
        rating_filter = st.multiselect(
            "Filter by Predicted Rating",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )
    with f2:
        neg_filter = st.selectbox(
            "Filter by Negative Triage Status",
            options=["All Reviews", "Negative Reviews Only (P >= 0.456)", "Non-Negative Reviews Only"],
            index=0
        )
    with f3:
        if selected_prod_col != "(None)" and selected_prod_col in results_df.columns:
            all_prods = ["All Products"] + sorted(results_df[selected_prod_col].dropna().unique().tolist())
            prod_filter = st.selectbox("Filter by Product / Brand", options=all_prods, index=0)
        else:
            prod_filter = "All Products"

    # Apply filters
    filtered_df = results_df[results_df["predicted_rating"].isin(rating_filter)].copy()
    if neg_filter == "Negative Reviews Only (P >= 0.456)":
        filtered_df = filtered_df[filtered_df["is_negative_predicted"] == True]
    elif neg_filter == "Non-Negative Reviews Only":
        filtered_df = filtered_df[filtered_df["is_negative_predicted"] == False]

    if prod_filter != "All Products" and selected_prod_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[selected_prod_col] == prod_filter]

    st.caption(f"Showing **{len(filtered_df):,}** of {total_batch:,} reviews matching filter criteria.")

    # Table View
    display_cols = [
        selected_text_col,
        "predicted_rating",
        "predicted_sentiment",
        "negative_probability",
        "is_negative_predicted",
        "priority_level",
        "regression_rating"
    ]
    if selected_rating_col != "(None / No Ground Truth)" and selected_rating_col in filtered_df.columns:
        display_cols.insert(1, selected_rating_col)
    if selected_prod_col != "(None)" and selected_prod_col in filtered_df.columns:
        display_cols.insert(0, selected_prod_col)

    st.dataframe(
        filtered_df[[c for c in display_cols if c in filtered_df.columns]],
        use_container_width=True,
        height=320
    )

    # Export CSV
    st.markdown("### 4. Export Analyzed Dataset")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_bytes,
        file_name=f"reviewiq_predictions_{len(filtered_df)}_reviews.csv",
        mime="text/csv",
        type="primary"
    )
