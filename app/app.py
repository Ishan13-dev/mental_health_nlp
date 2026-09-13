import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# --- PATH & MODEL IMPORT SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import MentalHealthClassifier

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="Mental Health NLP Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# --- SIDEBAR (Presets only) ---
with st.sidebar:
    st.markdown("### 💡 Quick Test Presets")
    if st.button("⚡ High Anxiety / Worry", use_container_width=True):
        st.session_state["input_text"] = "I've been feeling extremely overwhelmed and anxious lately. I can't sleep and my heart races at night."
        st.rerun()
    if st.button("🔋 Chronic Burnout", use_container_width=True):
        st.session_state["input_text"] = "Work has been draining all my energy. I feel detached, constantly exhausted, and unable to focus on simple tasks."
        st.rerun()
    if st.button("🌿 Neutral / Balanced", use_container_width=True):
        st.session_state["input_text"] = "Today was a balanced day. Managed to take a walk, finish my assignments, and read a few chapters of my book."
        st.rerun()

# --- HEADER ---
st.title("🧠 Mental Health NLP Analyzer")
st.write("Type how you are feeling below. The model will analyze the text and predict the category.")

# --- MODEL LOADER ---
@st.cache_resource
def get_classifier():
    return MentalHealthClassifier()

classifier = get_classifier()

# --- INPUT SECTION ---
st.text_area(
    "Express your current thoughts, feelings, or recent state of mind:",
    key="input_text",
    height=150,
    placeholder="Write freely (e.g., 'I feel overwhelmed by deadlines and can't focus on anything...')"
)

analyze_btn = st.button("✨ Analyze Text", type="primary")

# --- INFERENCE & VISUALIZATION ---
if analyze_btn:
    if not st.session_state["input_text"].strip():
        st.warning("⚠️ Please provide some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = classifier.predict(st.session_state["input_text"])

        top_pred = results[0]
        plot_df = pd.DataFrame(results)
        top_label = top_pred["label"]
        confidence_pct = top_pred["confidence"] * 100

        st.markdown("---")

        # Top Prediction Metric
        st.subheader("Top Prediction")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(label="Predicted Category", value=top_label, delta=f"{confidence_pct:.1f}% Confidence", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Probability Breakdown")
            color_palette = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#6366F1', '#14B8A6']
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=plot_df['label'],
                values=plot_df['confidence'],
                hole=0.65,
                textinfo='percent',
                textposition='inside',
                marker=dict(colors=color_palette[:len(plot_df)])
            )])
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(t=20, b=20, l=20, r=20),
                hovermode=False,
                annotations=[dict(text=f"<b>{top_label}</b><br>{confidence_pct:.1f}%", x=0.5, y=0.5, font=dict(size=14, color="#3B82F6"), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_chart2:
            st.subheader("Confidence Scores")
            
            fig_bar = px.bar(
                plot_df,
                x='confidence',
                y='label',
                orientation='h',
                text=plot_df['confidence'].apply(lambda v: f"{v * 100:.1f}%"),
                color='confidence',
                color_continuous_scale=['#8B5CF6', '#3B82F6']
            )
            fig_bar.update_traces(textposition='outside', cliponaxis=False)
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# --- DISCLAIMER SECTION ---
st.markdown("---")
st.warning(
    "**⚠️ Disclaimer:** This application is an educational/research prototype. "
    "It is not a medical diagnostic system. Model predictions should not be treated as professional medical advice. "
    "If you are in crisis, please contact a licensed mental health professional or a crisis helpline immediately."
)