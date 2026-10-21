"""
ReviewIQ — Page 7: Business Recommendations
Translates model outputs into actionable, evidence-based business operations
with assigned team ownership, priority levels, and concrete next steps.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.ui_helpers import render_sidebar, get_active_dataframe
from src.recommendations import generate_business_recommendations

st.set_page_config(
    page_title="Business Recommendations | ReviewIQ",
    page_icon="🎯",
    layout="wide"
)

artifacts = render_sidebar("Business Recommendations")

st.title("🎯 Actionable Business Recommendations")
st.markdown(
    "Operational action plans derived strictly from the active review dataset. "
    "Every recommendation is grounded in quantified evidence, assigned to an operational team, and labeled by rule vs. AI origin."
)

df = get_active_dataframe()

if df is None:
    st.info("No active dataset loaded in memory. Please load sample reviews or upload a CSV file.", icon="ℹ️")
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

# Generate grounded recommendations
recommendations = generate_business_recommendations(
    df,
    text_col=text_col,
    pred_rating_col="predicted_rating",
    neg_prob_col="negative_probability",
    product_col=prod_col
)

if not recommendations:
    st.info("No actionable recommendations could be derived from the current review sample.")
    st.stop()

# Priority count metrics
crit_cnt = sum(1 for r in recommendations if r["priority"] == "Critical")
high_cnt = sum(1 for r in recommendations if r["priority"] == "High")
med_cnt = sum(1 for r in recommendations if r["priority"] == "Medium")
low_cnt = sum(1 for r in recommendations if r["priority"] == "Low")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Critical Actions", f"{crit_cnt}")
with k2:
    st.metric("High-Priority Actions", f"{high_cnt}")
with k3:
    st.metric("Medium-Priority Actions", f"{med_cnt}")
with k4:
    st.metric("Total Strategic Items", f"{len(recommendations)}")

st.markdown("---")

# Filter controls
f_col1, f_col2 = st.columns(2)
with f_col1:
    priority_filter = st.multiselect(
        "Filter by Priority Level",
        options=["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )
with f_col2:
    type_filter = st.selectbox(
        "Filter by Diagnostic Origin",
        options=["All Types", "Rule-Based Operational Rules Only", "AI-Assisted Diagnostics Only"],
        index=0
    )

filtered_recs = [r for r in recommendations if r["priority"] in priority_filter]
if type_filter == "Rule-Based Operational Rules Only":
    filtered_recs = [r for r in filtered_recs if "Rule-Based" in r["type"]]
elif type_filter == "AI-Assisted Diagnostics Only":
    filtered_recs = [r for r in filtered_recs if "AI-Assisted" in r["type"]]

st.caption(f"Showing **{len(filtered_recs)}** recommendations based on active filters.")

# Render Recommendation Cards
for idx, rec in enumerate(filtered_recs):
    p_color = (
        "#DC2626" if rec["priority"] == "Critical"
        else ("#EA580C" if rec["priority"] == "High"
              else ("#D97706" if rec["priority"] == "Medium" else "#16A34A"))
    )

    badge_bg = (
        "#FEE2E2" if "Rule-Based" in rec["type"] else "#EFF6FF"
    )
    badge_text = (
        "#991B1B" if "Rule-Based" in rec["type"] else "#1E40AF"
    )

    st.markdown(
        f"""
        <div style="background-color: white; border: 1px solid #E2E8F0; border-left: 6px solid {p_color}; border-radius: 8px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span style="background-color: {badge_bg}; color: {badge_text}; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase;">
                        {rec['type']}
                    </span>
                    <h3 style="margin: 8px 0 4px 0; color: #0F172A; font-size: 1.3rem;">{rec['title']}</h3>
                    <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
                        Assigned Owner: <strong>{rec['owner']}</strong> • Impact Scope: <strong>{rec['percentage']}</strong>
                    </p>
                </div>
                <div style="text-align: right; margin-top: 6px;">
                    <span style="background-color: {p_color}1A; color: {p_color}; border: 1px solid {p_color}40; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 9999px;">
                        {rec['priority']} Priority
                    </span>
                </div>
            </div>
            <div style="margin-top: 16px; padding: 12px 16px; background-color: #F8FAFC; border-radius: 6px; border: 1px solid #EDF2F7;">
                <p style="margin: 0; font-size: 0.92rem; color: #334155;">
                    📊 <strong>Supporting Evidence:</strong> {rec['evidence']}
                </p>
            </div>
            <div style="margin-top: 14px;">
                <p style="margin: 0; font-size: 0.95rem; color: #1E293B; line-height: 1.5;">
                    🎯 <strong>Suggested Operational Next Step:</strong><br>{rec['suggested_action']}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Action Plan Export
st.subheader("📋 Export Operational Roadmap")
rec_export_df = pd.DataFrame([
    {
        "Priority": r["priority"],
        "Title": r["title"],
        "Diagnostic Type": r["type"],
        "Owner Team": r["owner"],
        "Evidence": r["evidence"],
        "Affected Scope": r["percentage"],
        "Recommended Next Step": r["suggested_action"]
    }
    for r in filtered_recs
])

st.download_button(
    "📥 Download Action Plan (CSV)",
    data=rec_export_df.to_csv(index=False).encode("utf-8"),
    file_name="reviewiq_operational_recommendations.csv",
    mime="text/csv",
    type="primary"
)
