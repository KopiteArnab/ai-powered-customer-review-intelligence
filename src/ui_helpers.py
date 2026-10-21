"""
ReviewIQ Shared UI Helpers and Layout Components.
Provides consistent styling, responsive cards, KPI metrics,
sidebar status, and Plotly charting configurations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional

from src.model_loader import load_artifacts, ModelArtifacts


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Applies a clean, modern, enterprise dashboard theme to Plotly figures."""
    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#1E293B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickcolor="#CBD5E1"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickcolor="#CBD5E1"
        )
    )
    return fig


def render_sidebar(current_page: str = "") -> ModelArtifacts:
    """Renders the standard ReviewIQ sidebar navigation, system status, and methodology."""
    artifacts = load_artifacts()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 15px 0;">
                <h2 style="margin: 0; color: #1E3A8A; font-size: 1.5rem; font-weight: 700;">ReviewIQ</h2>
                <p style="margin: 2px 0 0 0; color: #64748B; font-size: 0.85rem;">Customer Review Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Operational Status Card
        st.markdown("#### System Health & Models")
        if artifacts.is_operational():
            st.success("Core NLP Pipeline Operational", icon="✅")
        else:
            st.error("Model Artifacts Missing", icon="⚠️")

        with st.expander("Loaded Model Components", expanded=False):
            st.markdown(f"- **5-Class Classifier:** {'✅ Ready' if artifacts.loaded_status['classification_model'] else '❌ Missing'}")
            st.markdown(f"- **Negative Detector:** {'✅ Ready' if artifacts.loaded_status['negative_review_model'] else '❌ Missing'}")
            st.markdown(f"- **Ridge Regressor:** {'✅ Ready' if artifacts.loaded_status['regression_model'] else '❌ Missing'}")
            st.markdown(f"- **Ordinal Model:** {'✅ Ready' if artifacts.loaded_status['ordinal_model'] else '❌ Missing'}")
            st.markdown(f"- **Validation Reports:** {'✅ Loaded' if artifacts.loaded_status['evaluation_reports'] else '❌ Missing'}")
            st.caption(f"Triage Decision Threshold: **{artifacts.negative_threshold:.3f}**")

        st.divider()

        # Active Data Context
        if "active_data" in st.session_state and st.session_state["active_data"] is not None:
            active_df = st.session_state["active_data"]
            st.info(f"**Active Dataset:** {len(active_df):,} reviews loaded", icon="📊")
        else:
            st.caption("No batch dataset loaded yet. Visit Batch Analysis or load sample data.")

        # Methodology & Limitations Expander
        with st.expander("Methodology & Limitations", expanded=False):
            st.markdown(
                """
                **Evaluation Safeguards:**
                - Trained with **StratifiedGroupKFold** grouped by normalized review body (zero duplicate-group leakage across splits).
                - Decision threshold for negative reviews (**0.456**) was selected exclusively on **validation data** (maximizing F2 score).
                - Feature order and TF-IDF parameters are strictly preserved.
                
                **Key Technical Limitations:**
                - The models predict the star rating implied by review text only.
                - Text alone cannot verify physical product defect rates, delivery carrier tracking, or true user satisfaction.
                - Ground-truth evaluation metrics require genuine customer star ratings and are not generated for unverified uploads.
                """
            )

        st.divider()
        st.caption("ReviewIQ Production ML System • Portfolio Edition")

    return artifacts


def get_active_dataframe() -> Optional[pd.DataFrame]:
    """Retrieves the active analyzed DataFrame from session state if available."""
    if "active_data" in st.session_state and isinstance(st.session_state["active_data"], pd.DataFrame):
        return st.session_state["active_data"]
    return None
