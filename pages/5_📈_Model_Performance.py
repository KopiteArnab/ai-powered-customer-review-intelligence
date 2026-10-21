"""
ReviewIQ — Page 5: Model Performance & Comparison
Comprehensive model evaluation dashboard comparing 5-Class Classification,
Cumulative Ordinal Modeling, Ridge Regression, and baselines.
Clearly segregates historical notebook test results from live runtime evaluations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_helpers import render_sidebar, get_active_dataframe, apply_plotly_theme
from src.evaluation import (
    get_historical_benchmarks,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_two_star_error_breakdown
)

st.set_page_config(
    page_title="Model Performance | ReviewIQ",
    page_icon="📈",
    layout="wide"
)

artifacts = render_sidebar("Model Performance")

st.title("📈 Model Performance & Comparative Evaluation")
st.markdown(
    "Benchmark evaluation analyzing classification accuracy, balanced macro F1, "
    "numeric regression error (MAE/RMSE/R²), and class-specific failure modes."
)

tab_historical, tab_live = st.tabs(["🏛️ Historical Notebook Evaluation Results", "⚡ Live Dataset Evaluation"])

# ----------------- TAB 1: HISTORICAL BENCHMARKS -----------------
with tab_historical:
    st.subheader("Reference Evaluation Results (Held-Out Test Set: 6,010 Reviews)")
    st.caption("ℹ️ These metrics reflect the official evaluation conducted in the training notebook on the 20% held-out test partition.")

    benchmarks = get_historical_benchmarks(artifacts)

    # 1. Summary comparison table
    st.markdown("#### Model Architecture Comparison")
    summary_df = pd.DataFrame(benchmarks["models_summary"])
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # 2. Detailed Classification vs Regression Breakdown
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 1. Five-Class Rating Classification (Held-Out Test)")
        if artifacts.classification_report is not None:
            cr_df = artifacts.classification_report.copy()
            # Clean display of report
            st.dataframe(cr_df, use_container_width=True)
        else:
            st.info("Historical classification report CSV not available.")

        st.markdown(
            """
            **Classification Insights:**
            - **TF-IDF + Balanced Logistic Regression:** Achieves **69.90% accuracy** and **0.5289 Macro F1**.
            - Balanced class weighting significantly lifts recall on minority classes (1-Star recall: **83.39%**).
            - Majority class baseline achieves 68.49% accuracy but has a dismal Macro F1 of **0.1626**, proving the necessity of macro F1.
            """
        )

    with col_c2:
        st.markdown("#### 2. Rating Regression Metrics (Held-Out Test)")
        if artifacts.regression_comparison is not None:
            st.dataframe(artifacts.regression_comparison, use_container_width=True)
        else:
            st.info("Historical regression comparison CSV not available.")

        st.markdown(
            """
            **Regression Insights:**
            - **TF-IDF + Ridge Regression:** Yields an **MAE of 0.4654 stars** and **R² of 0.5708**.
            - On average, the model's numeric score deviates by less than half a star from customer ratings.
            - Outperforms the median baseline (MAE: 0.59 stars, R²: -0.28).
            """
        )

    st.markdown("---")

    # 3. Dedicated Two-Star Prediction Error Analysis
    st.markdown("#### 🔬 Deep Dive: Two-Star Prediction Error & Ordinal Modeling Trade-Off")
    c_two1, c_two2 = st.columns([1.2, 1])

    with c_two1:
        if artifacts.ordinal_classification_report is not None:
            st.markdown("##### Cumulative Ordinal Logistic Report")
            st.dataframe(artifacts.ordinal_classification_report, use_container_width=True)

    with c_two2:
        st.markdown("##### The Two-Star Dilemma")
        st.markdown(
            """
            **Why Two-Star Reviews are Difficult:**
            - **Severe Class Imbalance:** Two-star reviews make up only ~1.1% of the dataset (69 samples in test).
            - **Linguistic Ambiguity:** Two-star reviews often use mixed language ("Good concept, but broke quickly"), confusing linear decision boundaries.
            - **Ordinal vs Standard Trade-Off:** While cumulative ordinal logistic regression achieved higher top-line accuracy (**73.88%**), its two-star recall collapsed to only **10.14%** (predicting mostly 1-star or 5-star).
            - **Conclusion:** Standard balanced Logistic Regression is preferred for operational triage due to superior minority recall.
            """
        )

# ----------------- TAB 2: LIVE EVALUATION -----------------
with tab_live:
    st.subheader("⚡ Live Evaluation on Active Dataset")
    active_df = get_active_dataframe()

    if active_df is None:
        st.info("No active dataset loaded. Please upload a dataset with ground-truth ratings in Batch Analysis to run live evaluation.")
    elif "reviews_rating" not in active_df.columns or active_df["reviews_rating"].dropna().empty:
        st.warning("The active dataset does not contain an actual customer rating column (`reviews_rating`). Live evaluation metrics require ground-truth labels.")
        st.caption("Displaying inference outputs without performance scores.")
    else:
        # We have actual ratings in active_df!
        eval_df = active_df.dropna(subset=["reviews_rating", "predicted_rating"]).copy()
        eval_df["actual_star"] = pd.to_numeric(eval_df["reviews_rating"], errors="coerce")
        eval_df = eval_df[eval_df["actual_star"].between(1, 5)].copy()

        if len(eval_df) < 5:
            st.warning("Insufficient valid ratings (minimum 5 required) to compute evaluation metrics.")
        else:
            st.success(f"Running live evaluation against **{len(eval_df):,}** validated ground-truth ratings.", icon="🎯")

            clf_metrics = compute_classification_metrics(eval_df["actual_star"], eval_df["predicted_rating"])
            reg_metrics = compute_regression_metrics(
                eval_df["actual_star"],
                eval_df["regression_rating"].fillna(eval_df["predicted_rating"])
            )

            # Live Metrics Row
            lm1, lm2, lm3, lm4, lm5 = st.columns(5)
            with lm1:
                st.metric("Live Accuracy", f"{clf_metrics['accuracy']:.1%}")
            with lm2:
                st.metric("Balanced Accuracy", f"{clf_metrics['balanced_accuracy']:.1%}")
            with lm3:
                st.metric("Macro F1 Score", f"{clf_metrics['macro_f1']:.3f}")
            with lm4:
                st.metric("Regression MAE", f"{reg_metrics['mae']:.3f} ★")
            with lm5:
                st.metric("Regression R²", f"{reg_metrics['r2']:.3f}")

            st.markdown("---")

            # Confusion Matrix & Error Visual
            c_cm, c_scat = st.columns(2)

            with c_cm:
                st.markdown("##### Live 5-Class Confusion Matrix")
                cm = clf_metrics["confusion_matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual {i}★" for i in range(1, 6)],
                    columns=[f"Pred {i}★" for i in range(1, 6)]
                )
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Rating", y="Actual Rating", color="Count")
                )
                fig_cm = apply_plotly_theme(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=True)

            with c_scat:
                st.markdown("##### Actual vs. Predicted Distribution")
                comp_data = []
                for star in [1, 2, 3, 4, 5]:
                    act = int((eval_df["actual_star"] == star).sum())
                    pred = int((eval_df["predicted_rating"] == star).sum())
                    comp_data.append({"Star": f"{star} Star", "Count": act, "Type": "Actual Ground Truth"})
                    comp_data.append({"Star": f"{star} Star", "Count": pred, "Type": "Model Prediction"})

                fig_comp = px.bar(
                    pd.DataFrame(comp_data),
                    x="Star",
                    y="Count",
                    color="Type",
                    barmode="group",
                    color_discrete_map={"Actual Ground Truth": "#64748B", "Model Prediction": "#2563EB"}
                )
                fig_comp = apply_plotly_theme(fig_comp)
                st.plotly_chart(fig_comp, use_container_width=True)

            # Two-star live breakdown
            st.markdown("##### Live Two-Star Error Diagnostics")
            two_star_res = compute_two_star_error_breakdown(eval_df["actual_star"], eval_df["predicted_rating"])
            if two_star_res["two_star_count"] > 0:
                st.markdown(
                    f"Out of **{two_star_res['two_star_count']}** actual 2-star reviews in this dataset, "
                    f"the model correctly predicted 2 stars for **{two_star_res['two_star_recall']:.1%}** of them."
                )
                two_dist = [
                    {"Predicted Rating": f"{s} Star", "Reviews": d["count"], "Share": f"{d['share']}%"}
                    for s, d in two_star_res["distribution"].items()
                ]
                st.table(pd.DataFrame(two_dist))
            else:
                st.caption("No two-star reviews present in active dataset.")
