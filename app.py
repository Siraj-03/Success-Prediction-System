# ============================================================
#   Startup Success Prediction — Streamlit Web App
#   Module 7 Frontend
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for streamlit
import shap
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Startup Success Predictor",
    page_icon  = "🚀",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)


# ── Paths ─────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.join(BASE_DIR, "outputs") + os.sep
MODELS   = os.path.join(BASE_DIR, "models")  + os.sep

# ── Load Model & Data ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODELS + "best_classical_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS + "best_model_name.txt", "r") as f:
        name = f.read().strip()
    return model, name

@st.cache_data
def load_data():
    X_train = pd.read_csv(BASE + "X_train.csv")
    X_test  = pd.read_csv(BASE + "X_test.csv")
    y_test  = pd.read_csv(BASE + "y_test.csv").squeeze()
    return X_train, X_test, y_test

@st.cache_data
def load_shap_summary():
    path = BASE + "shap_feature_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ml_comparison():
    path = BASE + "ml_model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

model, model_name = load_model()
X_train, X_test, y_test = load_data()
feature_names = list(X_train.columns)
shap_summary  = load_shap_summary()
ml_comparison = load_ml_comparison()




# ── Custom CSS Styling ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .subtitle {
        text-align: center; color: #888;
        font-size: 1.1rem; margin-bottom: 2rem;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; font-size: 1.5rem; font-weight: 700;
        box-shadow: 0 4px 15px rgba(56,239,125,0.3);
    }
    .failure-box {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; font-size: 1.5rem; font-weight: 700;
        box-shadow: 0 4px 15px rgba(235,51,73,0.3);
    }
    .metric-card {
        background: #1e1e2e; padding: 20px;
        border-radius: 12px; text-align: center;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 Startup Success Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered prediction using Machine Learning, ANN & Deep Learning</div>',
            unsafe_allow_html=True)
st.markdown("---")


# ── Scaling parameters (from raw dataset statistics) ──────────
# These map raw user inputs to the standardized scale the model expects
SCALE_PARAMS = {
    "funding_total_usd":      {"mean": 12706538.38, "std": 13217344.21},
    "funding_rounds":         {"mean": 2.153,       "std": 1.237},
    "milestones":             {"mean": 1.733,       "std": 1.247},
    "relationships":          {"mean": 6.174,       "std": 4.520},
    "age_first_funding_year": {"mean": 2.529,       "std": 2.737},
    "age_last_funding_year":  {"mean": 3.975,       "std": 3.276},
    "has_VC":                 {"mean": 0.320,       "std": 0.467},
    "has_angel":              {"mean": 0.267,       "std": 0.443},
    "has_roundA":             {"mean": 0.495,       "std": 0.500},
    "has_roundB":             {"mean": 0.366,       "std": 0.482},
    "has_roundC":             {"mean": 0.157,       "std": 0.364},
    "has_roundD":             {"mean": 0.090,       "std": 0.286},
    "is_CA":                  {"mean": 0.486,       "std": 0.500},
    "is_NY":                  {"mean": 0.111,       "std": 0.315},
    "is_MA":                  {"mean": 0.087,       "std": 0.282},
    "is_TX":                  {"mean": 0.047,       "std": 0.212},
    "avg_participants":       {"mean": 2.243,       "std": 1.738},
    "log_funding":            {"mean": 15.380,      "std": 1.763},
}

def scale_val(raw, feature):
    """Scale a raw value using dataset mean/std."""
    if feature in SCALE_PARAMS:
        p = SCALE_PARAMS[feature]
        return (raw - p["mean"]) / p["std"]
    return raw

# ── Sidebar — User Inputs ─────────────────────────────────────
st.sidebar.header("📋 Enter Startup Details")
st.sidebar.markdown("Adjust the values below to match your startup profile.")

with st.sidebar:
    st.subheader("💰 Funding Info")
    funding_total   = st.slider("Total Funding (USD)", 0, 50_000_000, 1_000_000, step=50_000)
    funding_rounds  = st.slider("Number of Funding Rounds", 0, 15, 3)

    st.subheader("📅 Timeline")
    age_first_fund  = st.slider("Age at First Funding (years)", 0.0, 10.0, 1.5, step=0.5)
    age_last_fund   = st.slider("Age at Last Funding (years)",  0.0, 15.0, 3.0, step=0.5)

    st.subheader("🏆 Milestones & Network")
    milestones      = st.slider("Number of Milestones Reached", 0, 10, 3)
    relationships   = st.slider("Number of Relationships/Connections", 0, 100, 20)

    st.subheader("📍 Location")
    state           = st.selectbox("State", ["CA", "NY", "MA", "TX", "Other"])

    st.subheader("🏢 Company Profile")
    has_angel       = st.checkbox("Has Angel Investors", value=True)
    has_vc          = st.checkbox("Has VC Funding",      value=False)
    has_roundA      = st.checkbox("Has Round A Funding", value=True)
    has_roundB      = st.checkbox("Has Round B Funding", value=False)
    is_top_category = st.checkbox("Top Industry Category (Enterprise/Biotech)", value=False)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Success", use_container_width=True, type="primary")



# ── Build Input Vector ────────────────────────────────────────
def build_input_vector(feature_names):
    """
    Creates a dataframe row matching exact feature columns from training.
    Raw user inputs are scaled to match the standardized training data.
    Unknown features default to 0 (neutral/scaled mean).
    """
    row = {f: 0.0 for f in feature_names}

    import math
    log_fund = math.log(funding_total) if funding_total > 0 else 0

    # Map user inputs → raw values, then scale
    raw_mapping = {
        "funding_total_usd":      funding_total,
        "funding_rounds":         funding_rounds,
        "milestones":             milestones,
        "relationships":          relationships,
        "age_first_funding_year": age_first_fund,
        "age_last_funding_year":  age_last_fund,
        "is_CA": 1 if state == "CA" else 0,
        "is_NY": 1 if state == "NY" else 0,
        "is_MA": 1 if state == "MA" else 0,
        "is_TX": 1 if state == "TX" else 0,
        "has_angel":       int(has_angel),
        "has_VC":          int(has_vc),
        "has_roundA":      int(has_roundA),
        "has_roundB":      int(has_roundB),
        "avg_participants": max(1, funding_rounds),
        "log_funding":     log_fund,
    }

    for k, raw in raw_mapping.items():
        if k in row:
            row[k] = scale_val(raw, k)


    return pd.DataFrame([row])[feature_names]

input_df = build_input_vector(feature_names)




# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 SHAP Explanation",
    "📈 Model Comparison",
    "🗂️ Dataset Insights"
])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔮 Startup Success Prediction")

    if predict_btn:
        proba      = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= 0.5)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="success-box">
                    ✅ ACQUIRED / SUCCESS<br>
                    <span style="font-size:1rem;font-weight:400">
                    Confidence: {proba*100:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="failure-box">
                    ❌ CLOSED / FAILURE<br>
                    <span style="font-size:1rem;font-weight:400">
                    Confidence: {(1-proba)*100:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

        st.markdown("###")

        # Probability gauge
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Success Probability",  f"{proba*100:.1f}%")
        col_b.metric("Failure Probability",  f"{(1-proba)*100:.1f}%")
        col_c.metric("Model Used", model_name)

        # Probability bar
        st.markdown("#### Confidence Breakdown")
        st.progress(float(proba))
        st.caption(f"← Closed (0%) ——————————————————— Acquired (100%) →")

        # Input summary
        st.markdown("#### 📋 Your Input Summary")
        display_df = pd.DataFrame({
            "Feature": ["Funding (USD)", "Funding Rounds", "Milestones",
                        "Relationships", "State", "Angel Investor", "VC Funding"],
            "Value":   [f"${funding_total:,}", funding_rounds, milestones,
                        relationships, state, has_angel, has_vc]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Fill in your startup details in the sidebar and click **Predict Success**")
        st.markdown("#### 🎯 How to Use This App")
        cols = st.columns(3)
        cols[0].success("**Step 1:** Adjust sliders in the sidebar to match your startup")
        cols[1].warning("**Step 2:** Click the Predict button")
        cols[2].info("**Step 3:** Explore SHAP tab to understand the prediction")



# ════════════════════════════════════════════════════════
# TAB 2 — SHAP EXPLANATION
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Why Did the Model Predict That?")
    st.markdown("SHAP values show exactly which features pushed toward success or failure.")

    if shap_summary is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Top Features Driving Success")
            top_pos = shap_summary.nlargest(10, "Mean_SHAP")[["Feature","Mean_SHAP"]]
            top_pos["Mean_SHAP"] = top_pos["Mean_SHAP"].round(4)
            st.dataframe(top_pos, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 📉 Feature Impact Chart")
            fig, ax = plt.subplots(figsize=(7, 5))
            top10 = shap_summary.head(10).sort_values("Mean_SHAP")
            bars  = ax.barh(top10["Feature"], top10["Mean_SHAP"],
                            color=plt.cm.RdYlGn(
                                np.linspace(0.3, 0.9, len(top10))))
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Feature Importance via SHAP", fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Live SHAP for current input
    if predict_btn:
        st.markdown("---")
        st.markdown("#### 🔍 SHAP for YOUR Startup Input")
        with st.spinner("Calculating SHAP values for your input..."):
            try:
                explainer  = shap.TreeExplainer(model)
                sv_input   = explainer.shap_values(input_df)

                # Handle all possible SHAP output formats safely
                if isinstance(sv_input, list):
                    sv_single = np.array(sv_input[1]).flatten()
                elif hasattr(sv_input, 'values'):
                    sv_arr = np.array(sv_input.values)
                    if sv_arr.ndim == 3:
                        sv_single = sv_arr[0, :, 1]
                    elif sv_arr.ndim == 2:
                        sv_single = sv_arr[0, :]
                    else:
                        sv_single = sv_arr.flatten()
                else:
                    sv_arr = np.array(sv_input)
                    if sv_arr.ndim == 3:
                        sv_single = sv_arr[0, :, 1]
                    elif sv_arr.ndim == 2:
                        sv_single = sv_arr[0, :]
                    else:
                        sv_single = sv_arr.flatten()

                # Safety check — must match number of features
                sv_single = sv_single[:len(feature_names)]

                top_n      = min(10, len(feature_names))
                sorted_i   = np.argsort(np.abs(sv_single))[::-1][:top_n]
                vals       = sv_single[sorted_i]
                names      = [feature_names[i] for i in sorted_i]
                bar_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(range(top_n), vals[::-1], color=bar_colors[::-1])
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(names[::-1], fontsize=9)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel("← Pushes to Closed  |  Pushes to Acquired →")
                ax.set_title("Why the Model Predicted This for YOUR Startup", fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP live explanation unavailable: {e}")
    else:
        st.info("👈 Click Predict first to see a live SHAP explanation for your startup.")



# ════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 All Models Compared")

    if ml_comparison is not None:
        st.markdown("#### Classical ML Models")
        st.dataframe(ml_comparison.style.highlight_max(
            subset=["Accuracy","ROC-AUC"], color="#2ecc7155"),
            use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x     = np.arange(len(ml_comparison))
        width = 0.35
        ax.bar(x - width/2, ml_comparison["Accuracy"], width,
               label="Accuracy", color="#4C72B0")
        ax.bar(x + width/2, ml_comparison["ROC-AUC"],  width,
               label="ROC-AUC",  color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(ml_comparison.index, rotation=15, ha='right')
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Classical ML — Accuracy vs ROC-AUC", fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Run Module 3 first to generate ml_model_comparison.csv")

    # Deep learning results
    deep_path = BASE + "deep_learning_comparison.csv"
    if os.path.exists(deep_path):
        st.markdown("#### Deep Learning Models (CNN / LSTM / CNN-LSTM)")
        deep_df = pd.read_csv(deep_path, index_col=0)
        st.dataframe(deep_df.style.highlight_max(color="#2ecc7155"),
                     use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — DATASET INSIGHTS
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🗂️ Dataset Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Startups",  len(X_train) + len(X_test))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Test Samples",     len(X_test))
    col4.metric("Features Used",    len(feature_names))

    st.markdown("#### 🎯 Target Distribution in Test Set")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart
    counts = y_test.value_counts()
    axes[0].pie(counts.values, labels=["Acquired","Closed"],
                autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'],
                startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0].set_title("Test Set Class Distribution", fontweight='bold')

    # Feature distribution
    top_feat = shap_summary["Feature"].iloc[0] if shap_summary is not None else feature_names[0]
    axes[1].hist(X_test[top_feat], bins=25, color='#4C72B0', alpha=0.7, edgecolor='white')
    axes[1].set_title(f"Distribution of Top Feature: {top_feat}", fontweight='bold')
    axes[1].set_xlabel(top_feat)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature list
    st.markdown("#### 📋 All 42 Features Used")
    feat_df = pd.DataFrame({"Feature Name": feature_names,
                            "Index": range(len(feature_names))})
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=300)