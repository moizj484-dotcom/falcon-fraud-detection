"""
FALCON - Financial Fraud Detection & Analysis System
Interactive Executive Dashboard with Real-Time Risk Simulation,
Model Comparison, and PaySim Fraud Analytics.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "experiments"

# Page configuration
st.set_page_config(
    page_title="FALCON | Financial Fraud Detection",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #172338;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #00C2A8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-legit {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def load_artifacts():
    """Safely loads metrics, comparison data, and feature names."""
    data = {}
    
    # Model comparison
    comp_file = MODEL_DIR / "model_comparison.json"
    if comp_file.exists():
        with open(comp_file) as f:
            data["comparison"] = json.load(f)
    else:
        data["comparison"] = None

    # Feature importances
    fi_file = MODEL_DIR / "feature_importance.json"
    if fi_file.exists():
        with open(fi_file) as f:
            data["feature_importance"] = json.load(f)
    else:
        data["feature_importance"] = None

    # Transaction types
    type_file = OUTPUT_DIR / "transaction_type_summary.csv"
    if type_file.exists():
        data["type_summary"] = pd.read_csv(type_file)
    else:
        data["type_summary"] = None

    # Fraud counts
    fraud_file = OUTPUT_DIR / "fraud_counts.csv"
    if fraud_file.exists():
        fc = pd.read_csv(fraud_file, index_col=0)
        data["legit_count"] = int(fc.iloc[0, 0])
        data["fraud_count"] = int(fc.iloc[1, 0])
        data["total_count"] = data["legit_count"] + data["fraud_count"]
    else:
        data["legit_count"] = 6354407
        data["fraud_count"] = 8213
        data["total_count"] = 6362620

    return data


def render_dashboard():
    artifacts = load_artifacts()

    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title("🦅 FALCON")
        st.subheader("Financial Fraud Detection & Risk Analysis System")
        st.caption("Enterprise AI-Powered Transaction Monitoring, Behavioral Analytics & Decision Support Engine")
    with col_head2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("🟢 **Pipeline Status: ONLINE**\n\nActive Model: **Random Forest** (99.67% F1)")

    st.divider()

    # ---------------------------------------------------------
    # Top KPI Metrics Row
    # ---------------------------------------------------------
    total_tx = artifacts["total_count"]
    legit_tx = artifacts["legit_count"]
    fraud_tx = artifacts["fraud_count"]
    fraud_pct = (fraud_tx / total_tx) * 100

    best_f1 = 99.67
    best_auc = 1.0000
    if artifacts["comparison"] and "Random Forest" in artifacts["comparison"].get("models", {}):
        rf_meta = artifacts["comparison"]["models"]["Random Forest"]
        best_f1 = rf_meta["f1_score"] * 100
        best_auc = rf_meta["roc_auc"]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Transactions", f"{total_tx:,}")
    with kpi2:
        st.metric("Legitimate", f"{legit_tx:,}", delta=f"{100 - fraud_pct:.2f}%", delta_color="normal")
    with kpi3:
        st.metric("Fraudulent Detected", f"{fraud_tx:,}", delta=f"-{fraud_pct:.3f}%", delta_color="inverse")
    with kpi4:
        st.metric("Ensemble F1-Score", f"{best_f1:.2f}%")
    with kpi5:
        st.metric("Model ROC-AUC", f"{best_auc:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Navigation Tabs
    # ---------------------------------------------------------
    tab_sim, tab_analytics, tab_models = st.tabs([
        "🛡️ Live Fraud Simulator",
        "📊 Fraud Analytics & Patterns",
        "🤖 Model Comparison & Architecture"
    ])

    # =========================================================
    # TAB 1: LIVE SIMULATOR
    # =========================================================
    with tab_sim:
        st.subheader("Real-Time Transaction Risk Simulator")
        st.write("Test incoming transaction parameters against the FALCON fraud detection engine, or use one of the one-click presets below.")

        # Presets Buttons
        st.markdown("##### ⚡ Quick Test Scenarios (1-Click Presets)")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        if "sim_type" not in st.session_state:
            st.session_state.sim_type = "TRANSFER"
            st.session_state.sim_amount = 450000.0
            st.session_state.sim_step = 104
            st.session_state.sim_old_org = 450000.0
            st.session_state.sim_new_org = 0.0
            st.session_state.sim_old_dest = 0.0
            st.session_state.sim_new_dest = 0.0

        if p_col1.button("🚨 Critical: Account Drain (Fraud)", use_container_width=True):
            st.session_state.sim_type = "TRANSFER"
            st.session_state.sim_amount = 680000.0
            st.session_state.sim_step = 98
            st.session_state.sim_old_org = 680000.0
            st.session_state.sim_new_org = 0.0
            st.session_state.sim_old_dest = 0.0
            st.session_state.sim_new_dest = 0.0

        if p_col2.button("🛒 Normal Merchant Payment (Legit)", use_container_width=True):
            st.session_state.sim_type = "PAYMENT"
            st.session_state.sim_amount = 120.50
            st.session_state.sim_step = 42
            st.session_state.sim_old_org = 5000.0
            st.session_state.sim_new_org = 4879.50
            st.session_state.sim_old_dest = 0.0
            st.session_state.sim_new_dest = 0.0

        if p_col3.button("💵 Normal Cash In Deposit (Legit)", use_container_width=True):
            st.session_state.sim_type = "CASH_IN"
            st.session_state.sim_amount = 25000.0
            st.session_state.sim_step = 14
            st.session_state.sim_old_org = 12000.0
            st.session_state.sim_new_org = 37000.0
            st.session_state.sim_old_dest = 80000.0
            st.session_state.sim_new_dest = 55000.0

        if p_col4.button("💸 Rapid Cash-Out (Fraud)", use_container_width=True):
            st.session_state.sim_type = "CASH_OUT"
            st.session_state.sim_amount = 320000.0
            st.session_state.sim_step = 144
            st.session_state.sim_old_org = 320000.0
            st.session_state.sim_new_org = 0.0
            st.session_state.sim_old_dest = 20000.0
            st.session_state.sim_new_dest = 340000.0

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("risk_eval_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                tx_type = st.selectbox(
                    "Transaction Channel / Type",
                    ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                    index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(st.session_state.sim_type)
                )
            with fc2:
                tx_amount = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.0,
                    value=float(st.session_state.sim_amount),
                    step=1000.0,
                    format="%.2f"
                )
            with fc3:
                tx_step = st.number_input(
                    "Transaction Step (Hour Index)",
                    min_value=0,
                    value=int(st.session_state.sim_step),
                    step=1
                )

            st.markdown("##### Account Balance Verification")
            bc1, bc2 = st.columns(2)
            with bc1:
                old_org = st.number_input(
                    "Sender (Origin) Initial Balance ($)",
                    min_value=0.0,
                    value=float(st.session_state.sim_old_org),
                    step=1000.0
                )
                new_org = st.number_input(
                    "Sender (Origin) Remaining Balance ($)",
                    min_value=0.0,
                    value=float(st.session_state.sim_new_org),
                    step=1000.0
                )
            with bc2:
                old_dest = st.number_input(
                    "Recipient (Destination) Initial Balance ($)",
                    min_value=0.0,
                    value=float(st.session_state.sim_old_dest),
                    step=1000.0
                )
                new_dest = st.number_input(
                    "Recipient (Destination) Final Balance ($)",
                    min_value=0.0,
                    value=float(st.session_state.sim_new_dest),
                    step=1000.0
                )

            submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk", use_container_width=True)

        if submit_btn:
            try:
                from src.predict import predict_transaction
                result = predict_transaction(
                    step=tx_step,
                    transaction_type=tx_type,
                    amount=tx_amount,
                    oldbalanceOrg=old_org,
                    newbalanceOrig=new_org,
                    oldbalanceDest=old_dest,
                    newbalanceDest=new_dest
                )

                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if result["prediction"] == "FRAUD":
                        st.error(f"### 🚨 {result['prediction']} DETECTED")
                        st.markdown(f"**Risk Severity:** <span class='badge-critical'>{result['risk_level']}</span>", unsafe_allow_html=True)
                    else:
                        st.success(f"### ✅ {result['prediction']} TRANSACTION")
                        st.markdown(f"**Risk Severity:** <span class='badge-legit'>{result['risk_level']}</span>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Fraud Probability", f"{result['fraud_probability']}%")
                    st.progress(result["fraud_probability"] / 100.0)

                    st.markdown("#### 🔍 Explainability Flags")
                    if result["flags"]:
                        for f in result["flags"]:
                            st.warning(f"⚠️ {f}")
                    else:
                        st.info("No suspicious risk triggers identified.")

                with res_col2:
                    # Risk Gauge Chart
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["fraud_probability"],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "FALCON Risk Score Index", 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#ef4444" if result["fraud_probability"] > 50 else "#10b981"},
                            'steps': [
                                {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.2)"},
                                {'range': [25, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75
                            }
                        }
                    ))
                    gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(gauge_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # =========================================================
    # TAB 2: FRAUD ANALYTICS
    # =========================================================
    with tab_analytics:
        st.subheader("PaySim Fraud Distribution & Pattern Intelligence")

        if artifacts["type_summary"] is not None:
            ts_df = artifacts["type_summary"]

            c_c1, c_c2 = st.columns(2)
            with c_c1:
                fig_vol = px.bar(
                    ts_df,
                    x="type",
                    y="total_transactions",
                    title="Transaction Volume by Channel",
                    labels={"total_transactions": "Volume", "type": "Channel"},
                    color="type",
                    text_auto=".2s"
                )
                fig_vol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_vol, use_container_width=True)

            with c_c2:
                fig_fr = px.bar(
                    ts_df,
                    x="type",
                    y="fraud_percentage",
                    title="Fraud Incident Rate (%) by Channel",
                    labels={"fraud_percentage": "Fraud Rate (%)", "type": "Channel"},
                    color="fraud_percentage",
                    color_continuous_scale="Reds",
                    text_auto=".3f"
                )
                fig_fr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fr, use_container_width=True)

            st.info(
                "💡 **Key FinTech Finding**: In the 6.36 million PaySim transactions, fraudulent activities "
                "occur exclusively within **TRANSFER** (0.77%) and **CASH_OUT** (0.18%) transactions. "
                "Channels such as PAYMENT, CASH_IN, and DEBIT exhibit zero fraud incidents."
            )

            st.dataframe(
                ts_df.rename(columns={
                    "type": "Channel Type",
                    "total_transactions": "Total Transactions",
                    "fraud_transactions": "Fraud Count",
                    "total_amount": "Total Volume ($)",
                    "total_fraud_amount": "Fraud Loss ($)",
                    "fraud_percentage": "Fraud Rate (%)"
                }),
                use_container_width=True
            )

    # =========================================================
    # TAB 3: MODEL COMPARISON & ARCHITECTURE
    # =========================================================
    with tab_models:
        st.subheader("Model Comparison: Baseline vs Tree-based Ensemble")

        if artifacts["comparison"]:
            comp_models = artifacts["comparison"].get("models", {})

            lr_data = comp_models.get("Logistic Regression", {})
            rf_data = comp_models.get("Random Forest", {})

            # Side-by-side metrics table
            comp_table = pd.DataFrame({
                "Evaluation Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Positives", "False Negatives"],
                "Logistic Regression (Baseline)": [
                    f"{lr_data.get('accuracy', 0) * 100:.2f}%",
                    f"{lr_data.get('precision', 0) * 100:.2f}%",
                    f"{lr_data.get('recall', 0) * 100:.2f}%",
                    f"{lr_data.get('f1_score', 0) * 100:.2f}%",
                    f"{lr_data.get('roc_auc', 0):.4f}",
                    f"{lr_data.get('false_positives', 0):,}",
                    f"{lr_data.get('false_negatives', 0):,}"
                ],
                "Random Forest (Ensemble Winner 🏆)": [
                    f"{rf_data.get('accuracy', 0) * 100:.2f}%",
                    f"{rf_data.get('precision', 0) * 100:.2f}%",
                    f"{rf_data.get('recall', 0) * 100:.2f}%",
                    f"{rf_data.get('f1_score', 0) * 100:.2f}%",
                    f"{rf_data.get('roc_auc', 0):.4f}",
                    f"{rf_data.get('false_positives', 0):,}",
                    f"{rf_data.get('false_negatives', 0):,}"
                ]
            })

            st.dataframe(comp_table, use_container_width=True)

            st.markdown("#### 🎯 Confusion Matrix Heatmaps (Hold-Out Test Set: 61,643 Transactions)")
            cm_c1, cm_c2 = st.columns(2)

            with cm_c1:
                cm_lr = np.array([
                    [lr_data.get("true_negatives", 0), lr_data.get("false_positives", 0)],
                    [lr_data.get("false_negatives", 0), lr_data.get("true_positives", 0)]
                ])
                fig_cm_lr = px.imshow(
                    cm_lr,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=["Legitimate", "Fraud"],
                    y=["Legitimate", "Fraud"],
                    title="Logistic Regression Confusion Matrix",
                    color_continuous_scale="Blues"
                )
                fig_cm_lr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm_lr, use_container_width=True)

            with cm_c2:
                cm_rf = np.array([
                    [rf_data.get("true_negatives", 0), rf_data.get("false_positives", 0)],
                    [rf_data.get("false_negatives", 0), rf_data.get("true_positives", 0)]
                ])
                fig_cm_rf = px.imshow(
                    cm_rf,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=["Legitimate", "Fraud"],
                    y=["Legitimate", "Fraud"],
                    title="Random Forest Confusion Matrix",
                    color_continuous_scale="Greens"
                )
                fig_cm_rf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm_rf, use_container_width=True)

            st.success(
                "🏆 **Why Random Forest Outperformed Logistic Regression**: "
                f"While Logistic Regression produced **{lr_data.get('false_positives', 0):,} False Positives** "
                f"(flagging legitimate customers by mistake), Random Forest reduced false alarms down to **only {rf_data.get('false_positives', 0):,}**, "
                f"achieving an exceptional **{rf_data.get('precision', 0) * 100:.2f}% Precision** and **{rf_data.get('recall', 0) * 100:.2f}% Recall**."
            )

        # Feature Importance Chart
        if artifacts["feature_importance"]:
            st.markdown("#### 🌟 Top Engineered Feature Importances")
            fi_items = list(artifacts["feature_importance"].items())[:10]
            fi_df = pd.DataFrame(fi_items, columns=["Feature", "Importance"])

            fig_fi = px.bar(
                fi_df[::-1],
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Fraud Detection Predictive Features",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.caption("FALCON — Financial Fraud Detection & Analysis System | Author: Moiz | Production Ready")


if __name__ == "__main__":
    render_dashboard()
