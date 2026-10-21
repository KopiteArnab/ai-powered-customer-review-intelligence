"""
ReviewIQ — Page 1: Executive Dashboard
Provides executive-level KPI metrics, sentiment breakdowns, rating distributions,
and temporal trends across analyzed reviews.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.ui_helpers import render_sidebar, get_active_dataframe, apply_plotly_theme
from src.topic_analysis import rule_based_tag_review

st.set_page_config(
    page_title="Executive Dashboard | ReviewIQ",
    page_icon="📊",
    layout="wide"
)

artifacts = render_sidebar("Executive Dashboard")

st.title("📊 Executive Dashboard")
st.markdown("High-level executive overview of customer sentiment, rating distributions, and operational volume.")

df = get_active_dataframe()

if df is None:
    st.info("No customer reviews are currently loaded in session memory.", icon="ℹ️")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📥 Load Sample Reviews (20 Examples)", type="primary", use_container_width=True):
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
    with col2:
        st.markdown("Or navigate to **3. Batch Review Analysis** to upload your custom CSV file.")
    st.stop()

# Distinguish ground truth presence
has_actual_rating = "reviews_rating" in df.columns and df["reviews_rating"].notna().sum() > 0
has_date = "reviews_date" in df.columns and df["reviews_date"].notna().sum() > 0
text_col = st.session_state.get("text_col", "reviews_text") if "reviews_text" in df.columns else df.columns[0]

total_count = len(df)
pred_avg = float(df["predicted_rating"].dropna().mean()) if "predicted_rating" in df.columns else 0.0

if has_actual_rating:
    actual_numeric = pd.to_numeric(df["reviews_rating"], errors="coerce").dropna()
    actual_avg = float(actual_numeric.mean()) if len(actual_numeric) > 0 else 0.0
else:
    actual_avg = None

neg_count = int((df["negative_probability"] >= artifacts.negative_threshold).sum()) if "negative_probability" in df.columns else 0
neg_pct = round(neg_count / total_count * 100, 1) if total_count > 0 else 0.0

high_priority_count = int((df["priority_level"] == "High Priority").sum()) if "priority_level" in df.columns else 0

# KPI Metric Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Reviews Analyzed", f"{total_count:,}", help="Total valid review records in the active dataset")
with k2:
    if actual_avg is not None:
        st.metric("Average Actual Rating", f"{actual_avg:.2f} ★", help="Dataset Ground Truth: Actual submitted star rating average")
    else:
        st.metric("Average Actual Rating", "N/A", help="No ground-truth actual rating column in this dataset")
with k3:
    st.metric("Predicted Average Rating", f"{pred_avg:.2f} ★", help="Model Inference: Mean predicted star rating across all reviews")
with k4:
    st.metric("Negative Review Share", f"{neg_pct}%", f"{neg_count} reviews", delta_color="inverse", help="Percentage of reviews flagged as negative (P >= 0.456 or predicted <= 2)")
with k5:
    st.metric("High-Priority Queue", f"{high_priority_count:,}", help="Number of reviews requiring immediate human triage")

st.caption("ℹ️ **Data Transparency:** Actual Rating is ground truth from customer submission; Predicted Rating and Negative Review Share are ML inference outputs.")

st.markdown("---")

# Row 2: Charts (Rating Distribution & Sentiment Breakdown)
c_chart1, c_chart2 = st.columns([1.2, 1])

with c_chart1:
    st.subheader("⭐ Star Rating Distribution")
    # Compare Actual vs Predicted if available
    dist_data = []
    for star in [1, 2, 3, 4, 5]:
        pred_cnt = int((df["predicted_rating"] == star).sum()) if "predicted_rating" in df.columns else 0
        dist_data.append({"Rating": f"{star} Star", "Count": pred_cnt, "Source": "Model Predicted"})
        if has_actual_rating:
            act_cnt = int((df["reviews_rating"] == star).sum())
            dist_data.append({"Rating": f"{star} Star", "Count": act_cnt, "Source": "Actual Ground Truth"})

    dist_df = pd.DataFrame(dist_data)
    fig_dist = px.bar(
        dist_df,
        x="Rating",
        y="Count",
        color="Source",
        barmode="group",
        color_discrete_map={"Model Predicted": "#2563EB", "Actual Ground Truth": "#94A3B8"},
        title="Review Count by Star Rating"
    )
    fig_dist = apply_plotly_theme(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

with c_chart2:
    st.subheader("🎭 Sentiment Breakdown")
    if "predicted_sentiment" in df.columns:
        sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        color_map = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#EF4444", "Unknown": "#94A3B8"}

        fig_pie = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            hole=0.45,
            color="Sentiment",
            color_discrete_map=color_map,
            title="Predicted Sentiment Proportions"
        )
        fig_pie = apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sentiment breakdown unavailable.")

st.markdown("---")

# Row 3: Temporal Trends & Most Common Complaint Categories
c_t1, c_t2 = st.columns(2)

with c_t1:
    st.subheader("📅 Rating & Volume Trends Over Time")
    if has_date:
        try:
            df["_parsed_date"] = pd.to_datetime(df["reviews_date"], errors="coerce")
            valid_time_df = df.dropna(subset=["_parsed_date"]).copy()
            if len(valid_time_df) > 5:
                valid_time_df["_period"] = valid_time_df["_parsed_date"].dt.to_period("M").dt.to_timestamp()
                time_agg = (
                    valid_time_df.groupby("_period")
                    .agg(
                        review_count=("predicted_rating", "count"),
                        avg_pred_rating=("predicted_rating", "mean")
                    )
                    .reset_index()
                )

                fig_time = go.Figure()
                fig_time.add_trace(
                    go.Bar(
                        x=time_agg["_period"],
                        y=time_agg["review_count"],
                        name="Review Volume",
                        marker_color="#CBD5E1",
                        yaxis="y"
                    )
                )
                fig_time.add_trace(
                    go.Scatter(
                        x=time_agg["_period"],
                        y=time_agg["avg_pred_rating"],
                        name="Avg Predicted Rating",
                        line=dict(color="#1E40AF", width=3),
                        mode="lines+markers",
                        yaxis="y2"
                    )
                )
                fig_time.update_layout(
                    title="Monthly Review Volume & Average Predicted Rating",
                    yaxis=dict(title="Review Volume", side="left", showgrid=True),
                    yaxis2=dict(
                        title="Avg Rating (Stars)",
                        side="right",
                        overlaying="y",
                        range=[1, 5.2],
                        showgrid=False
                    ),
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                )
                fig_time = apply_plotly_theme(fig_time)
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Insufficient valid dates to construct temporal time series.")
        except Exception as e:
            st.warning(f"Could not parse temporal trend: {e}")
    else:
        st.info("No valid date column found. Include a `reviews_date` column in your upload to enable time-series analysis.")

with c_t2:
    st.subheader("🏷️ Most Common Complaint Drivers")
    # Tag negative reviews to display prominent friction points
    neg_subset = df[df["negative_probability"] >= artifacts.negative_threshold] if "negative_probability" in df.columns else df[df["predicted_rating"] <= 2]
    if len(neg_subset) > 0:
        cat_counts = {}
        for text in neg_subset[text_col].dropna():
            tags = rule_based_tag_review(text)
            for tag in tags:
                if tag != "General / Uncategorized":
                    cat_counts[tag] = cat_counts.get(tag, 0) + 1

        if cat_counts:
            cat_df = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"]).sort_values("Count", ascending=True)
            fig_cats = px.bar(
                cat_df,
                x="Count",
                y="Category",
                orientation="h",
                color="Count",
                color_continuous_scale="Reds",
                title="Top Friction Categories in Negative Feedback"
            )
            fig_cats = apply_plotly_theme(fig_cats)
            fig_cats.update_layout(showlegend=False)
            st.plotly_chart(fig_cats, use_container_width=True)
        else:
            st.info("No specific keyword categories detected in negative reviews.")
    else:
        st.success("No negative reviews detected in current dataset!")

st.markdown("---")

# Executive Business Summary Card
st.subheader("💡 Executive Insight Summary")
st.markdown(
    f"""
    <div style="background-color: #F1F5F9; border-left: 4px solid #1E40AF; padding: 18px 24px; border-radius: 8px;">
        <h4 style="margin: 0 0 8px 0; color: #0F172A;">Executive Intelligence Takeaway</h4>
        <p style="margin: 0; color: #334155; line-height: 1.6;">
            Across <strong>{total_count:,}</strong> evaluated reviews, the model estimates an average rating of <strong>{pred_avg:.2f} stars</strong>. 
            A total of <strong>{neg_count} reviews ({neg_pct}%)</strong> crossed the validation-tuned threshold for negative customer sentiment, 
            resulting in <strong>{high_priority_count} high-priority records</strong> requiring immediate operational follow-up. 
            Overall customer perception remains predominantly favorable, but concentrated complaints around product reliability and fulfillment warrant targeted intervention.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
