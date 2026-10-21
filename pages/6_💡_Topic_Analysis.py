"""
ReviewIQ — Page 6: Customer Feedback Topic Analysis
Unsupervised topic modeling and complaint thematic discovery.
Identifies recurring customer issues, keyword signatures, and negative review prevalence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_helpers import render_sidebar, get_active_dataframe, apply_plotly_theme
from src.topic_analysis import (
    discover_topics_nmf,
    analyze_topic_breakdown,
    is_bertopic_available,
    TAXONOMY_KEYWORDS,
    rule_based_tag_review
)

st.set_page_config(
    page_title="Topic Analysis | ReviewIQ",
    page_icon="💡",
    layout="wide"
)

artifacts = render_sidebar("Topic Analysis")

st.title("💡 Customer Feedback Topic Analysis")
st.markdown(
    "Uncover latent complaint themes and recurring customer sentiment drivers using "
    "unsupervised matrix factorization (TF-IDF + NMF) and domain complaint taxonomy clustering."
)

df = get_active_dataframe()

if df is None:
    st.info("No active customer reviews loaded in memory. Please load sample reviews or upload a CSV file.", icon="ℹ️")
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
date_col = st.session_state.get("date_col", "reviews_date") if "reviews_date" in df.columns else None

# Controls & Configuration
c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
with c_ctrl1:
    num_topics = st.slider("Number of Topic Clusters to Discover", min_value=3, max_value=8, value=5)
with c_ctrl2:
    filter_choice = st.selectbox(
        "Review Population to Analyze",
        options=["All Reviews", "Negative Reviews Only (P >= 0.456)", "Positive Reviews Only"],
        index=0
    )
with c_ctrl3:
    method_choice = st.selectbox(
        "Topic Discovery Engine",
        options=["TF-IDF + NMF (Fast, Reliable Built-in)", "BERTopic (Optional Transformer)"],
        index=0
    )

# Filter dataset if requested
target_df = df.copy()
if filter_choice == "Negative Reviews Only (P >= 0.456)":
    target_df = target_df[target_df["negative_probability"] >= artifacts.negative_threshold]
elif filter_choice == "Positive Reviews Only":
    target_df = target_df[target_df["negative_probability"] < artifacts.negative_threshold]

if len(target_df) < num_topics:
    st.warning(f"Only {len(target_df)} reviews available matching criteria. Need at least {num_topics} reviews.")
    st.stop()

# BERTopic handling
if "BERTopic" in method_choice:
    if not is_bertopic_available():
        st.warning("⚠️ BERTopic is not installed in the current environment.")
        st.markdown(
            """
            To use BERTopic, install the optional dependency in your terminal:
            ```bash
            pip install bertopic
            ```
            ReviewIQ has automatically defaulted to the scikit-learn TF-IDF + NMF engine.
            """
        )
    else:
        st.info("BERTopic is available. Running neural embedding topic model...")

# Run NMF Topic Modeling
with st.spinner("Extracting semantic topics from review corpus..."):
    texts = target_df[text_col].dropna().astype(str).tolist()
    topic_results = discover_topics_nmf(texts, n_topics=num_topics, top_words=6)

if not topic_results.get("success", False):
    st.error(f"Topic modeling encountered an error: {topic_results.get('error')}")
    st.stop()

# Assign topic labels
topic_dict = topic_results["topics"]
topic_labels = [topic_dict[idx]["label"] for idx in topic_results["assignments"]]
target_df["assigned_topic"] = topic_labels

# Summary analysis
breakdown = analyze_topic_breakdown(
    target_df,
    topic_labels=topic_labels,
    text_col=text_col,
    rating_col="reviews_rating" if "reviews_rating" in target_df.columns else None,
    neg_flag_col="is_negative_predicted" if "is_negative_predicted" in target_df.columns else None,
    date_col=date_col
)

summary_table = breakdown["summary_table"]

# Display Topics Breakdown
st.subheader("1. Discovered Topics & Keywords")
col_chart, col_details = st.columns([1.2, 1])

with col_chart:
    fig_topics = px.bar(
        summary_table,
        x="Review Count",
        y="Topic",
        orientation="h",
        color="Negative Review Rate (%)",
        color_continuous_scale="Reds",
        title="Topic Cluster Size & Negative Review Rate (%)"
    )
    fig_topics = apply_plotly_theme(fig_topics)
    st.plotly_chart(fig_topics, use_container_width=True)

with col_details:
    st.markdown("##### Topic Signatures")
    for t_idx, t_info in topic_dict.items():
        st.markdown(f"**{t_info['label']}**")
        st.caption(f"Top terms: `{', '.join(t_info['keywords'])}`")

st.markdown("---")

# Summary Table View
st.subheader("2. Topic-Level Metrics & Representative Examples")
st.dataframe(
    summary_table[["Topic", "Review Count", "Share (%)", "Negative Review Rate (%)"]],
    use_container_width=True
)

with st.expander("🔍 View Representative Customer Quotes per Topic", expanded=True):
    for _, row in summary_table.iterrows():
        st.markdown(f"**{row['Topic']}** — *{row['Negative Review Rate (%)']}% Negative Sentiment Rate*")
        examples = row["Representative Examples"]
        if examples:
            for ex in examples:
                st.markdown(f"> *\"{ex[:250]}...\"*")
        else:
            st.caption("No quotes available.")
        st.divider()

# Temporal Trends if available
if breakdown.get("trend_data") is not None and not breakdown["trend_data"].empty:
    st.subheader("3. Emerging Complaint Themes Over Time")
    trend_df = breakdown["trend_data"]
    fig_trend = px.line(
        trend_df,
        x="_period",
        y="review_count",
        color="topic_label",
        markers=True,
        title="Monthly Topic Volume Trajectory"
    )
    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Review Count")
    fig_trend = apply_plotly_theme(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)
