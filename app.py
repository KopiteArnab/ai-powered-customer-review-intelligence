"""
ReviewIQ — AI-Powered Customer Review Intelligence System
Main Application Entrypoint.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.model_loader import load_artifacts
from src.ui_helpers import render_sidebar, get_active_dataframe
from src.prediction import predict_batch

st.set_page_config(
    page_title="ReviewIQ — Customer Review Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

artifacts = render_sidebar("Home")

# Header Section
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #2563EB 100%); padding: 32px 36px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <span style="background-color: rgba(255, 255, 255, 0.2); padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Enterprise NLP & Operations Dashboard</span>
        <h1 style="color: white; margin: 12px 0 8px 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em;">ReviewIQ</h1>
        <p style="color: #E2E8F0; margin: 0; font-size: 1.1rem; max-width: 850px; line-height: 1.5;">
            An end-to-end customer review rating prediction, negative-sentiment detection, and automated complaint triage system built with production ML safeguards.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Status and Overview Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        label="Rating Classifier",
        value="5-Class Logistic",
        help="TF-IDF + class-weighted Logistic Regression predicting 1 to 5 stars"
    )
with c2:
    st.metric(
        label="Triage Decision Threshold",
        value=f"{artifacts.negative_threshold:.3f}",
        help="Validation-selected threshold for maximum F2 complaint recall"
    )
with c3:
    st.metric(
        label="Rating Regressor",
        value="Ridge Regression",
        help="Numeric rating estimator with MAE ~0.465 stars and R² ~0.571"
    )
with c4:
    active_df = get_active_dataframe()
    st.metric(
        label="Active Reviews in Memory",
        value=f"{len(active_df):,}" if active_df is not None else "0 (Sample Available)",
        help="Total customer reviews currently loaded in session memory"
    )

st.markdown("---")

# Quick Start / Data Preload Section
st.subheader("🚀 Quick Start: Load Review Dataset")
st.markdown("Get started instantly by loading the pre-configured sample reviews or full dataset into session state.")

col_a, col_b, col_c = st.columns([1, 1, 1.5])
with col_a:
    if st.button("📥 Load Sample Reviews (20 Scenarios)", use_container_width=True, type="primary"):
        sample_path = Path("data/sample_reviews.csv")
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path)
            with st.spinner("Processing sample reviews through trained ML pipeline..."):
                df_analyzed = predict_batch(
                    df_sample,
                    text_col="reviews_text",
                    title_col="reviews_title",
                    artifacts=artifacts
                )
                st.session_state["active_data"] = df_analyzed
                st.session_state["text_col"] = "reviews_text"
                st.session_state["title_col"] = "reviews_title"
                st.session_state["rating_col"] = "reviews_rating"
                st.session_state["product_col"] = "name"
                st.session_state["date_col"] = "reviews_date"
            st.success("Loaded and analyzed 20 sample reviews! Navigate to any dashboard page.", icon="🎉")
            st.rerun()
        else:
            st.error("Sample dataset file not found at data/sample_reviews.csv.")

with col_b:
    if st.button("📊 Load Benchmark Subset (500 Reviews)", use_container_width=True):
        bench_path = Path("reviews.csv")
        if bench_path.exists():
            with st.spinner("Sampling and running predictions on 500 dataset reviews..."):
                df_raw = pd.read_csv(bench_path, nrows=500)
                df_analyzed = predict_batch(
                    df_raw,
                    text_col="reviews_text",
                    title_col="reviews_title",
                    artifacts=artifacts
                )
                st.session_state["active_data"] = df_analyzed
                st.session_state["text_col"] = "reviews_text"
                st.session_state["title_col"] = "reviews_title"
                st.session_state["rating_col"] = "reviews_rating"
                st.session_state["product_col"] = "name"
                st.session_state["date_col"] = "reviews_date"
            st.success("Loaded and analyzed 500 benchmark reviews!", icon="🚀")
            st.rerun()
        else:
            st.error("reviews.csv dataset not found in workspace.")

with col_c:
    if active_df is not None:
        st.success(f"**Session Active:** {len(active_df):,} reviews analyzed. Explore the pages in the sidebar.", icon="✅")
    else:
        st.info("💡 You can also upload any custom customer review CSV on **Page 3: Batch Analysis**.")

st.markdown("---")

# Architecture & Modules
st.subheader("📑 System Architecture & Application Modules")

modules = [
    ("📊 Page 1: Executive Dashboard", "High-level summary KPIs, rating distributions, sentiment breakdown, and temporal volume trends."),
    ("🔍 Page 2: Single Review Analysis", "Live interactive inference for individual reviews with multi-model comparison, risk triage, and support SLAs."),
    ("📁 Page 3: Batch Review Analysis", "Flexible CSV uploader with column auto-detection, schema validation, batch inference, and results export."),
    ("🚨 Page 4: Negative Review & Complaint Triage", "Operational triage queue with dynamic decision threshold adjustment and live precision-recall trade-off metrics."),
    ("📈 Page 5: Model Performance & Comparison", "Rigorous evaluation benchmarking 5-class classification, cumulative ordinal modeling, Ridge regression, and 2-star confusion analysis."),
    ("💡 Page 6: Customer Feedback Topic Analysis", "NLP unsupervised topic discovery (NMF) and complaint taxonomy clustering to surface root causes."),
    ("🎯 Page 7: Business Recommendations", "Evidence-based, prioritized action plans for QA, Logistics, Merchandising, and Customer Care operations.")
]

for title, desc in modules:
    st.markdown(f"**{title}** — {desc}")

st.markdown("---")

# Production ML Safeguards
st.subheader("🛡️ Machine Learning Safeguards & Design Decisions")
c_s1, c_s2 = st.columns(2)
with c_s1:
    st.markdown(
        """
        - **Zero Duplicate-Group Leakage:** Evaluations group normalized review bodies via `StratifiedGroupKFold` so identical review templates never cross train and test sets.
        - **Validation-Only Threshold Selection:** The negative triage threshold (0.456) was optimized for F2 recall on validation data alone; test data was touched strictly once for final reporting.
        """
    )
with c_s2:
    st.markdown(
        """
        - **No Inference Retraining:** The application strictly applies pre-trained frozen vectorizers and estimators during inference, never refitting on user-uploaded test data.
        - **Honest Metric Transparency:** Ground-truth evaluation metrics are never synthesized. Historical notebook benchmarks are explicitly segregated from live runtime metrics.
        """
    )
