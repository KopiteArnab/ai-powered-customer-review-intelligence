"""
ReviewIQ — Page 2: Single Review Analysis
Interactive single-review inference engine with multi-model comparison,
risk prioritization, and customer support action guidance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui_helpers import render_sidebar, apply_plotly_theme
from src.prediction import predict_single_review, predict_zero_shot_hf

st.set_page_config(
    page_title="Single Review Analysis | ReviewIQ",
    page_icon="🔍",
    layout="wide"
)

artifacts = render_sidebar("Single Review")

st.title("🔍 Single Review Analysis")
st.markdown(
    "Enter or select a customer review to predict its star rating, detect negative sentiment risk, "
    "and generate immediate customer support triage recommendations."
)

SAMPLE_REVIEWS = {
    "Select a pre-filled sample review...": {
        "title": "",
        "text": ""
    },
    "⭐⭐⭐⭐⭐ 5-Star: Enthusiastic Product Praise": {
        "title": "Exceptional sound and battery life",
        "text": "These headphones exceeded my expectations. The active noise cancellation blocks out all office chatter, and the battery easily lasts 30+ hours on a single charge. Super comfortable padding too."
    },
    "⭐ 1-Star: Severe Defect & Safety Concern": {
        "title": "Severe defect and burning smell",
        "text": "The charger became scalding hot within 15 minutes of plugging in and gave off a dangerous burning plastic odor. Sparked when unplugged. Complete fire hazard!"
    },
    "⭐⭐ 2-Star: Substantial Quality Flaw": {
        "title": "Broken pump dispenser and half leaked",
        "text": "The lotion itself smells nice, but the bottle arrived cracked at the nozzle and leaked all over the shipping box. Difficult to use without a working dispenser."
    },
    "⭐⭐⭐ 3-Star: Mixed Experience": {
        "title": "Decent sound but uncomfortable",
        "text": "Audio quality and bass are good for the price, but the headband clamp pressure causes headaches after an hour of use. Mixed feelings overall."
    },
    "⭐ 1-Star: Poor Customer Service & Refused Return": {
        "title": "Terrible customer support",
        "text": "Item arrived broken and customer support was completely unhelpful. They refused to process a refund or replacement and hung up. Worst shopping experience."
    }
}

# Sample Selector
sample_choice = st.selectbox(
    "💡 Or choose a sample scenario:",
    options=list(SAMPLE_REVIEWS.keys()),
    index=0
)

# Text inputs
col_t1, col_t2 = st.columns([1, 2.5])
with col_t1:
    default_title = SAMPLE_REVIEWS[sample_choice]["title"]
    input_title = st.text_input("Review Title (Optional)", value=default_title, placeholder="e.g. Great sound quality")
    
    with st.expander("⚙️ Decision Threshold Setting", expanded=False):
        custom_threshold = st.slider(
            "Negative Triage Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(artifacts.negative_threshold),
            step=0.01,
            help="Reviews with P(Negative) >= this threshold are classified as Negative and flagged for High Priority."
        )
        st.caption(f"Default validation-selected threshold: `{artifacts.negative_threshold:.3f}`")

with col_t2:
    default_text = SAMPLE_REVIEWS[sample_choice]["text"]
    input_text = st.text_area(
        "Customer Review Text",
        value=default_text,
        height=140,
        placeholder="Paste full customer review text here..."
    )

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    analyze_btn = st.button("🚀 Analyze Review", type="primary", use_container_width=True)

if analyze_btn or (input_text and sample_choice != "Select a pre-filled sample review..."):
    if not input_text.strip():
        st.warning("Please enter review text to analyze.")
        st.stop()

    with st.spinner("Analyzing text through multi-model pipeline..."):
        results = predict_single_review(
            review_text=input_text,
            review_title=input_title,
            artifacts=artifacts,
            negative_threshold=custom_threshold
        )

    if not results.get("is_valid", False):
        st.error(results.get("error", "Failed to analyze review."))
        st.stop()

    st.markdown("---")

    # Primary Prediction Banner
    star_rating = results["predicted_rating"]
    sentiment = results["predicted_sentiment"]
    neg_prob = results["negative_probability"]
    priority = results["priority_level"]
    confidence = results.get("model_confidence", 0.0)
    is_neg = results["is_negative_review"]

    # Color coded summary card
    card_color = "#DC2626" if priority == "High Priority" else ("#D97706" if priority == "Medium Priority" else "#16A34A")
    star_str = "★" * star_rating + "☆" * (5 - star_rating)

    st.markdown(
        f"""
        <div style="background-color: white; border: 1px solid #E2E8F0; border-left: 6px solid {card_color}; padding: 22px 28px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Primary Model Prediction</span>
                    <h2 style="margin: 4px 0; color: #0F172A; font-size: 2rem;">{star_rating} Stars <span style="color: #F59E0B; font-size: 1.6rem;">{star_str}</span></h2>
                    <p style="margin: 0; color: #475569; font-size: 1rem;">Predicted Sentiment: <strong>{sentiment}</strong> • Top-Class Probability: <strong>{confidence:.1%}</strong></p>
                </div>
                <div style="text-align: right; margin-top: 8px;">
                    <span style="display: inline-block; background-color: {card_color}1A; color: {card_color}; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.95rem; border: 1px solid {card_color}40;">
                        {priority}
                    </span>
                    <p style="margin: 6px 0 0 0; color: #64748B; font-size: 0.85rem;">Negative Review Probability: <strong>{neg_prob:.1%}</strong></p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Detailed KPI Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Predicted Rating",
            value=f"{star_rating} / 5 Stars",
            help="Predicted by TF-IDF + Balanced Logistic Regression"
        )
    with m2:
        st.metric(
            label="Negative Probability",
            value=f"{neg_prob:.1%}",
            delta="Flagged Negative" if is_neg else "Non-Negative",
            delta_color="inverse" if is_neg else "normal",
            help=f"Trained binary classifier output. Threshold = {custom_threshold:.3f}"
        )
    with m3:
        reg_val = results.get("regression_rating")
        st.metric(
            label="Ridge Rating Estimate",
            value=f"{reg_val:.2f} Stars" if reg_val is not None else "N/A",
            help="Continuous rating prediction from TF-IDF + Ridge Regression"
        )
    with m4:
        st.metric(
            label="Triage Priority",
            value=priority,
            help="Rule-based triage level combining negative probability and predicted rating"
        )

    # Row 2: Model Comparison & Probability Breakdown
    st.markdown("### 📊 Multi-Model Output Breakdown")
    col_chart, col_models = st.columns([1.3, 1])

    with col_chart:
        st.markdown("##### 5-Class Prediction Probability Distribution")
        probs = results.get("rating_probabilities", {})
        if probs:
            prob_df = pd.DataFrame([
                {"Rating": f"{star} Star", "Probability": p, "Is_Top": star == star_rating}
                for star, p in probs.items()
            ])
            fig_probs = px.bar(
                prob_df,
                x="Rating",
                y="Probability",
                text="Probability",
                color="Is_Top",
                color_discrete_map={True: "#1E40AF", False: "#CBD5E1"},
                range_y=[0, 1.05]
            )
            fig_probs.update_traces(texttemplate="%{y:.1%}", textposition="outside")
            fig_probs = apply_plotly_theme(fig_probs)
            fig_probs.update_layout(showlegend=False, height=280)
            st.plotly_chart(fig_probs, use_container_width=True)

    with col_models:
        st.markdown("##### Model Agreement Summary")
        comparison_rows = [
            {"Model Architecture": "5-Class Balanced Logistic", "Output": f"{star_rating} Stars ({sentiment})"},
            {"Model Architecture": "Ridge Rating Regressor", "Output": f"{reg_val:.2f} Stars" if reg_val else "N/A"},
            {"Model Architecture": "Binary Negative Detector", "Output": f"{neg_prob:.1%} P(Neg) → {'Negative' if is_neg else 'Non-Negative'}"}
        ]
        if "ordinal_predicted_rating" in results:
            comparison_rows.append({
                "Model Architecture": "Cumulative Ordinal Logistic",
                "Output": f"{results['ordinal_predicted_rating']} Stars"
            })
        st.table(pd.DataFrame(comparison_rows))

    # Support Guidance and SLA Card
    st.markdown("### 🎯 Recommended Customer Support Action")
    guidance = results.get("guidance", {})
    if guidance:
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px;">
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px;">
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Recommended Next Action</span>
                        <h4 style="margin: 4px 0 8px 0; color: #0F172A;">{guidance.get('action', 'Standard Support')}</h4>
                        <p style="margin: 0; color: #334155; font-size: 0.95rem; line-height: 1.5;">{guidance.get('suggested_response', '')}</p>
                    </div>
                    <div style="width: 220px;">
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Target SLA</span>
                        <p style="margin: 4px 0 12px 0; font-weight: 700; color: #1E40AF; font-size: 1.1rem;">{guidance.get('sla', 'N/A')}</p>
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Assigned Team</span>
                        <p style="margin: 4px 0 0 0; font-weight: 600; color: #334155;">{guidance.get('routing', 'General Queue')}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Optional Zero-Shot Hugging Face Comparison Expander
    st.markdown("---")
    with st.expander("🤖 Optional: Run Hugging Face Zero-Shot Classifier (facebook/bart-large-mnli)", expanded=False):
        st.markdown(
            "Test zero-shot natural language inference without dataset-specific fine-tuning. "
            "Requires active internet connectivity and downloads BART weights (~1.6GB) on first run."
        )
        hf_token_input = st.text_input(
            "Hugging Face Token (Optional - Leave blank for public models)",
            type="password",
            help="Loaded from HF_TOKEN environment variable or secrets if left blank."
        )
        if st.button("Run Zero-Shot Classification", key="btn_zero_shot"):
            with st.spinner("Executing Hugging Face zero-shot classification pipeline..."):
                zs_res = predict_zero_shot_hf(input_text, hf_token=hf_token_input if hf_token_input else None)
            if zs_res.get("success"):
                st.success(f"**Zero-Shot Predicted Rating:** {zs_res['predicted_rating']} Stars (Confidence: {zs_res['confidence']:.1%})")
                st.caption(f"Matched Candidate Label: `{zs_res['best_label']}`")
                st.json(zs_res["all_scores"])
            else:
                st.warning(f"Zero-shot execution unavailable: {zs_res.get('error', 'Unknown error')}")
                st.caption(zs_res.get("message", ""))

    # Limitations Callout
    st.caption(f"⚠️ **Model Limitations:** {results['limitations']}")
