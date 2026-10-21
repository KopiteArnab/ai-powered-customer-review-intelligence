"""
ReviewIQ — Page 4: Negative Review & Complaint Triage
Dedicated complaint triage queue powered by the binary classifier.
Provides real-time decision threshold tuning and precision-recall trade-off analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_helpers import render_sidebar, get_active_dataframe, apply_plotly_theme
from src.evaluation import compute_negative_triage_metrics

st.set_page_config(
    page_title="Negative Review Triage | ReviewIQ",
    page_icon="🚨",
    layout="wide"
)

artifacts = render_sidebar("Negative Review Triage")

st.title("🚨 Negative Review & Complaint Triage")
st.markdown(
    "Operational triage dashboard designed to identify, prioritize, and route customer complaints. "
    "Negative reviews are defined as **1- or 2-star ratings**; 3- to 5-star reviews are classified as non-negative."
)

df = get_active_dataframe()

if df is None:
    st.info("No active dataset loaded in memory. Load sample reviews or upload a CSV file.", icon="ℹ️")
    if st.button("📥 Load Sample Reviews (20 Examples)", type="primary"):
        from src.prediction import predict_batch
        sample_df = pd.read_csv("data/sample_reviews.csv")
        analyzed = predict_batch(sample_df, text_col="reviews_text", title_col="reviews_title", artifacts=artifacts)
        st.session_state["active_data"] = analyzed
        st.session_state["text_col"] = "reviews_text"
        st.session_state["title_col"] = "reviews_title"
        st.session_state["rating_col"] = "reviews_rating"
        st.session_state["product_col"] = "name"
        st.session_state["date_col"] = "reviews_date"
        st.rerun()
    st.stop()

text_col = st.session_state.get("text_col", "reviews_text") if "reviews_text" in df.columns else df.columns[0]
prod_col = st.session_state.get("product_col", "name") if "name" in df.columns else None
has_ground_truth = "reviews_rating" in df.columns and df["reviews_rating"].notna().sum() >= 5

# Threshold Controls
col_slider, col_expl = st.columns([1.5, 2])
with col_slider:
    triage_threshold = st.slider(
        "Decision Threshold for Negative Review Classification",
        min_value=0.10,
        max_value=0.90,
        value=float(artifacts.negative_threshold),
        step=0.01,
        help="Lower threshold = Higher recall (catches more complaints, but increases false alarms). Higher threshold = Higher precision."
    )
    st.caption(f"Default project validation threshold: **{artifacts.negative_threshold:.3f}** (Optimized for maximum F2 recall)")

with col_expl:
    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 6px; font-size: 0.88rem; color: #334155;">
            <strong>Threshold Trade-Off Mechanics:</strong><br>
            • <strong>Lowering threshold (&lt; 0.456):</strong> Prioritizes recall. Minimizes missed customer complaints at the cost of reviewing more neutral/borderline reviews.<br>
            • <strong>Raising threshold (&gt; 0.456):</strong> Prioritizes precision. Only high-conviction complaints are escalated, but some 2-star complaints may be overlooked.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Recalculate negative flags at current slider threshold
df_triage = df.copy()
if "negative_probability" in df_triage.columns:
    df_triage["is_neg_at_threshold"] = df_triage["negative_probability"] >= triage_threshold
else:
    df_triage["is_neg_at_threshold"] = df_triage["predicted_rating"] <= 2

total_count = len(df_triage)
flagged_count = int(df_triage["is_neg_at_threshold"].sum())
flagged_share = round(flagged_count / total_count * 100, 1) if total_count > 0 else 0.0

# KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Reviews in Queue", f"{total_count:,}")
with k2:
    st.metric("Flagged Negative Reviews", f"{flagged_count:,}", f"{flagged_share}%", delta_color="inverse")
with k3:
    high_priority_count = int((df_triage["is_neg_at_threshold"] & (df_triage["predicted_rating"] <= 2)).sum())
    st.metric("High-Priority Critical Items", f"{high_priority_count:,}")
with k4:
    st.metric("Current Operating Threshold", f"{triage_threshold:.3f}")

# Ground-truth evaluation if ratings available
if has_ground_truth:
    st.markdown("### 🎯 Live Model Evaluation at Current Threshold")
    metrics = compute_negative_triage_metrics(
        y_true_ratings=df_triage["reviews_rating"],
        neg_probabilities=df_triage["negative_probability"].fillna(0.0),
        threshold=triage_threshold
    )

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric("Precision", f"{metrics['precision']:.1%}", help="Share of flagged reviews that are actually 1-2 stars")
    with e2:
        st.metric("Recall (Sensitivity)", f"{metrics['recall']:.1%}", help="Share of actual 1-2 star reviews that were successfully caught")
    with e3:
        st.metric("F1 Score", f"{metrics['f1']:.3f}", help="Harmonic mean of precision and recall")
    with e4:
        st.metric("F2 Score (Recall-Weighted)", f"{metrics['f2']:.3f}", help="Recall-weighted metric used for project validation")
    with e5:
        st.metric("Average Precision (PR-AUC)", f"{metrics['average_precision']:.3f}")

    # Confusion matrix visual
    c_cm1, c_cm2 = st.columns([1, 1.5])
    with c_cm1:
        cm_data = [
            {"Actual": "Non-Negative (3–5★)", "Predicted": "Non-Negative", "Count": metrics["true_negatives"]},
            {"Actual": "Non-Negative (3–5★)", "Predicted": "Flagged Negative", "Count": metrics["false_positives"]},
            {"Actual": "Negative (1–2★)", "Predicted": "Non-Negative", "Count": metrics["false_negatives"]},
            {"Actual": "Negative (1–2★)", "Predicted": "Flagged Negative", "Count": metrics["true_positives"]}
        ]
        cm_df = pd.DataFrame(cm_data).pivot(index="Actual", columns="Predicted", values="Count").fillna(0)
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix (Threshold = {triage_threshold:.3f})"
        )
        fig_cm = apply_plotly_theme(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_cm2:
        st.markdown(
            f"""
            ##### Operational Interpretation
            - **True Positives ({metrics['true_positives']}):** Real customer complaints correctly escalated for human intervention.
            - **False Positives ({metrics['false_positives']}):** Non-negative reviews flagged (creates additional agent review time).
            - **False Negatives ({metrics['false_negatives']}):** Severe complaints that slipped past the filter (critical customer defection risk).
            - **True Negatives ({metrics['true_negatives']}):** Satisfied reviews safely processed without wasting support capacity.
            """
        )
else:
    st.info("ℹ️ Ground-truth ratings are not present in this dataset. Performance metrics (Precision, Recall, F1, F2) require actual customer star ratings.")

st.markdown("---")

# Triage Queue Filtering and Display
st.subheader("📋 Escalation Review Queue")

q_f1, q_f2, q_f3 = st.columns(3)
with q_f1:
    queue_filter = st.selectbox(
        "Queue Filter",
        options=["Flagged Negative Reviews Only", "All Reviews"],
        index=0
    )
with q_f2:
    priority_filter = st.multiselect(
        "Priority Level",
        options=["High Priority", "Medium Priority", "Low Priority"],
        default=["High Priority", "Medium Priority"]
    )
with q_f3:
    if prod_col and prod_col in df_triage.columns:
        prod_opts = ["All Products"] + sorted(df_triage[prod_col].dropna().unique().tolist())
        selected_prod = st.selectbox("Product Filter", options=prod_opts, index=0)
    else:
        selected_prod = "All Products"

queue_df = df_triage.copy()
if queue_filter == "Flagged Negative Reviews Only":
    queue_df = queue_df[queue_df["is_neg_at_threshold"] == True]

if priority_filter:
    queue_df = queue_df[queue_df["priority_level"].isin(priority_filter)]

if selected_prod != "All Products" and prod_col in queue_df.columns:
    queue_df = queue_df[queue_df[prod_col] == selected_prod]

# Sort by highest negative probability
if "negative_probability" in queue_df.columns:
    queue_df = queue_df.sort_values("negative_probability", ascending=False)

st.caption(f"Showing **{len(queue_df):,}** reviews matching escalation criteria.")

cols_to_show = [text_col, "predicted_rating", "negative_probability", "priority_level"]
if "reviews_rating" in queue_df.columns:
    cols_to_show.insert(1, "reviews_rating")
if prod_col and prod_col in queue_df.columns:
    cols_to_show.insert(0, prod_col)

st.dataframe(
    queue_df[[c for c in cols_to_show if c in queue_df.columns]],
    use_container_width=True,
    height=340
)

# Queue Export
csv_queue = queue_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Export High-Priority Escalation Queue (CSV)",
    data=csv_queue,
    file_name=f"triage_queue_threshold_{triage_threshold:.3f}.csv",
    mime="text/csv",
    type="primary"
)
