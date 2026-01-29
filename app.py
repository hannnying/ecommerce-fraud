import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.config import API_PORT, BACKEND_HOST
from streamlit_autorefresh import st_autorefresh

BACKEND_URL = f"http://{BACKEND_HOST}:{API_PORT}"

# Page configuration
st.set_page_config(
    page_title="E-commerce Fraud Detection - Model Performance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Fraud Detection Model Performance Monitor")

# ============================================================================
# Sidebar Configuration
# ============================================================================
st.sidebar.header("⚙️ Settings")

# Refresh settings
REFRESH_INTERVAL = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=1,
    max_value=60,
    value=5,
    step=1
)

# Fraud threshold settings
st.sidebar.subheader("🎯 Fraud Detection")
fraud_threshold = st.sidebar.slider(
    "Fraud probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Transactions above this probability are classified as fraud"
)

# Rolling window settings
st.sidebar.subheader("📊 Rolling Window")
window_sizes = st.sidebar.multiselect(
    "Window sizes to display",
    options=[1000, 5000, 10000, 20000],
    default=[1000, 5000, 10000],
    help="Select which rolling window sizes to monitor"
)

# Auto-refresh
st_autorefresh(interval=REFRESH_INTERVAL * 1000, limit=None)

# ============================================================================
# Helper Functions
# ============================================================================
def fetch_data(endpoint, params=None, default=None):
    """Fetch data from backend with error handling."""
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return default
    except Exception as e:
        st.sidebar.warning(f"API Error ({endpoint}): {str(e)[:50]}")
        return default

# ============================================================================
# Fetch Data from Backend
# ============================================================================
# Cumulative evaluation
eval_cumulative = fetch_data("/evaluate", params={"threshold": fraud_threshold}, default=None)

# Rolling window evaluations
eval_rolling = {}
for window_size in window_sizes:
    data = fetch_data(
        "/evaluate/rolling",
        params={"window_size": window_size, "threshold": fraud_threshold},
        default=None
    )
    if data:
        eval_rolling[window_size] = data

# Drift monitoring
drift_window_size = min(window_sizes) if window_sizes else 5000
drift_data = fetch_data(
    "/drift/check",
    params={"window_size": drift_window_size, "threshold": 0.2},
    default=None
)

# ============================================================================
# SECTION 0: Fraud Rate Overview
# ============================================================================
st.header("📊 Fraud Rate Overview")

col_overview1, col_overview2 = st.columns(2)

with col_overview1:
    if eval_cumulative and eval_cumulative.get('count', 0) > 0:
        cumulative_actual_rate = eval_cumulative.get('actual_fraud_rate', 0) * 100 if 'actual_fraud_rate' in eval_cumulative else None
        cumulative_predicted_rate = eval_cumulative.get('predicted_fraud_rate', 0) * 100 if 'predicted_fraud_rate' in eval_cumulative else None

        st.subheader("🗂️ Cumulative (All Transactions)")

        if cumulative_actual_rate is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Actual Fraud Rate",
                    value=f"{cumulative_actual_rate:.2f}%",
                    help="Ground truth fraud rate across all labeled transactions"
                )
            with col_b:
                if cumulative_predicted_rate is not None:
                    delta = cumulative_predicted_rate - cumulative_actual_rate
                    st.metric(
                        label="Predicted Fraud Rate",
                        value=f"{cumulative_predicted_rate:.2f}%",
                        delta=f"{delta:+.2f}%",
                        help="Model's predicted fraud rate across all transactions"
                    )

        st.caption(f"Based on {eval_cumulative.get('count', 0):,} labeled transactions")
    else:
        st.info("⏳ Waiting for labeled transactions...")

with col_overview2:
    if eval_rolling and window_sizes:
        # Use the smallest window size as "current" (most recent)
        current_window_size = min(window_sizes)
        current_data = eval_rolling.get(current_window_size)

        if current_data and current_data.get('count', 0) > 0:
            current_actual_rate = current_data.get('actual_fraud_rate', 0) * 100
            current_predicted_rate = current_data.get('predicted_fraud_rate', 0) * 100

            st.subheader(f"⚡ Current (Last {current_window_size:,} Txns)")

            col_c, col_d = st.columns(2)
            with col_c:
                # Calculate delta vs cumulative if available
                delta_vs_cumulative = None
                if eval_cumulative and 'actual_fraud_rate' in eval_cumulative:
                    cumulative_rate = eval_cumulative.get('actual_fraud_rate', 0) * 100
                    delta_vs_cumulative = current_actual_rate - cumulative_rate

                st.metric(
                    label="Actual Fraud Rate",
                    value=f"{current_actual_rate:.2f}%",
                    delta=f"{delta_vs_cumulative:+.2f}% vs cumulative" if delta_vs_cumulative is not None else None,
                    delta_color="inverse" if delta_vs_cumulative and delta_vs_cumulative > 0 else "normal",
                    help="Ground truth fraud rate in the most recent window"
                )

            with col_d:
                delta_pred_vs_actual = current_predicted_rate - current_actual_rate
                st.metric(
                    label="Predicted Fraud Rate",
                    value=f"{current_predicted_rate:.2f}%",
                    delta=f"{delta_pred_vs_actual:+.2f}%",
                    help="Model's predicted fraud rate in the most recent window"
                )

            st.caption(f"Based on last {current_data.get('count', 0):,} labeled transactions")
        else:
            st.info(f"⏳ Waiting for {current_window_size:,} labeled transactions...")
    else:
        st.info("⏳ Select window sizes in sidebar to enable current fraud rate monitoring")

st.divider()

# ============================================================================
# SECTION: Data Drift Detection
# ============================================================================
st.header("🔍 Data Drift Detection")
st.caption("Monitors changes in feature distributions compared to training data")

if drift_data and "drift_results" in drift_data and drift_data["drift_results"]:
    # Alert if significant drift detected
    num_drifted = drift_data.get("num_features_drifted", 0)
    total_features = drift_data.get("total_features_monitored", 0)
    should_retrain = drift_data.get("should_retrain", False)

    if should_retrain:
        st.error(f"⚠️ **Significant drift detected!** {num_drifted} out of {total_features} features have drifted. Model retraining recommended.")
    elif num_drifted > 0:
        st.warning(f"⚡ Moderate drift detected in {num_drifted} feature(s). Continue monitoring.")
    else:
        st.success(f"✅ No significant drift detected across {total_features} monitored features.")

    # Summary metrics
    col_drift1, col_drift2, col_drift3 = st.columns(3)

    with col_drift1:
        st.metric(
            label="Features Monitored",
            value=total_features
        )

    with col_drift2:
        st.metric(
            label="Features Drifted",
            value=num_drifted,
            delta=f"{(num_drifted/total_features*100):.1f}%" if total_features > 0 else "0%",
            delta_color="inverse"
        )

    with col_drift3:
        st.metric(
            label="Window Size",
            value=f"{drift_data.get('window_size', 0):,}"
        )

    # Drift results table
    st.subheader("Drift Scores by Feature")

    # Prepare data for display
    drift_results = drift_data["drift_results"]
    drift_table_data = []

    for feature, result in sorted(drift_results.items(), key=lambda x: x[1]["drift_score"], reverse=True):
        drift_table_data.append({
            "Feature": feature,
            "Type": result["type"].capitalize(),
            "Drift Score": result["drift_score"],
            "Status": "🔴 Drifted" if result["drifted"] else "🟢 OK",
            "Drifted": result["drifted"]
        })

    df_drift = pd.DataFrame(drift_table_data)

    # Color-code the table
    def highlight_drift(row):
        if row["Drifted"]:
            return ['background-color: #ffcccc'] * len(row)  # Light red
        else:
            return ['background-color: #ccffcc'] * len(row)  # Light green

    # Display styled dataframe
    st.dataframe(
        df_drift.drop(columns=["Drifted"]).style.apply(highlight_drift, axis=1).format({
            "Drift Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Drift score visualization
    st.subheader("Drift Score Visualization")

    fig_drift = go.Figure()

    # Add bars for each feature
    colors = ['#e74c3c' if row["Drifted"] else '#2ecc71' for _, row in df_drift.iterrows()]

    fig_drift.add_trace(go.Bar(
        x=df_drift["Feature"],
        y=df_drift["Drift Score"],
        marker_color=colors,
        text=df_drift["Drift Score"].round(4),
        textposition='outside'
    ))

    # Add threshold line
    threshold = drift_data.get("threshold", 0.2)
    fig_drift.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="right"
    )

    fig_drift.update_layout(
        xaxis_title="Feature",
        yaxis_title="Jensen-Shannon Divergence",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_drift, use_container_width=True)

    # Additional info
    with st.expander("ℹ️ About Drift Detection"):
        st.markdown(f"""
        **Jensen-Shannon Divergence** measures the difference between two probability distributions:
        - **0.0** = Identical distributions (no drift)
        - **1.0** = Completely different distributions (maximum drift)

        **Interpretation:**
        - JS < 0.1: No significant drift
        - 0.1 ≤ JS < 0.2: Moderate drift (monitor)
        - JS ≥ 0.2: Significant drift (consider retraining)

        **Current Threshold:** {threshold}

        **Baseline:** Training data from run `{drift_data.get('run_id', 'unknown')}`
        """)

elif drift_data and "message" in drift_data:
    st.info(f"ℹ️ {drift_data['message']}")
else:
    st.warning("⏳ Drift monitoring data not available. Ensure model is trained and transactions are streaming.")

st.divider()

# ============================================================================
# SECTION 1: Cumulative Model Performance (All Labeled Transactions)
# ============================================================================
st.header("📈 Cumulative Model Performance")
st.caption("Performance metrics calculated on **all labeled transactions** since deployment")

if eval_cumulative and eval_cumulative.get('count', 0) > 0:
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="📊 Labeled Txns",
            value=f"{eval_cumulative.get('count', 0):,}"
        )

    with col2:
        st.metric(
            label="🎯 PR-AUC",
            value=f"{eval_cumulative.get('pr_auc', 0):.3f}"
        )

    with col3:
        st.metric(
            label="✓ Precision",
            value=f"{eval_cumulative.get('precision', 0):.3f}"
        )

    with col4:
        st.metric(
            label="↻ Recall",
            value=f"{eval_cumulative.get('recall', 0):.3f}"
        )

    with col5:
        st.metric(
            label="⚖️ F1 Score",
            value=f"{eval_cumulative.get('f1', 0):.3f}"
        )

    with col6:
        st.metric(
            label="📍 Accuracy",
            value=f"{eval_cumulative.get('accuracy', 0):.3f}"
        )

    st.info(f"ℹ️ Threshold: {fraud_threshold:.2f} | Evaluated on all {eval_cumulative.get('count', 0):,} labeled transactions")

else:
    st.warning("⏳ Cumulative evaluation metrics will appear once labeled transactions are available in Redis.")

st.divider()

# ============================================================================
# SECTION 2: Rolling Window Performance (Recent Transactions)
# ============================================================================
st.header("🔄 Rolling Window Performance")
st.caption("Performance metrics calculated on **recent labeled transactions** (sliding window)")

if not eval_rolling:
    st.warning("⏳ Rolling window metrics will appear once enough labeled transactions are available.")
else:
    # Display metrics for each window size
    for window_size in sorted(eval_rolling.keys()):
        data = eval_rolling[window_size]

        if data.get('count', 0) == 0:
            st.info(f"Window {window_size:,}: Waiting for {window_size:,} labeled transactions...")
            continue

        st.subheader(f"Window: Last {window_size:,} Labeled Transactions")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="📊 Count",
                value=f"{data.get('count', 0):,}"
            )

        with col2:
            st.metric(
                label="🎯 PR-AUC",
                value=f"{data.get('pr_auc', 0):.3f}"
            )

        with col3:
            st.metric(
                label="✓ Precision",
                value=f"{data.get('precision', 0):.3f}"
            )

        with col4:
            st.metric(
                label="↻ Recall",
                value=f"{data.get('recall', 0):.3f}"
            )

        with col5:
            st.metric(
                label="⚖️ F1 Score",
                value=f"{data.get('f1', 0):.3f}"
            )

        # Fraud rate comparison
        col_fraud1, col_fraud2 = st.columns(2)

        with col_fraud1:
            actual_rate = data.get('actual_fraud_rate', 0) * 100
            st.metric(
                label="📉 Actual Fraud Rate",
                value=f"{actual_rate:.2f}%",
                help="Percentage of transactions that are actually fraudulent (ground truth)"
            )

        with col_fraud2:
            predicted_rate = data.get('predicted_fraud_rate', 0) * 100
            delta = predicted_rate - actual_rate
            st.metric(
                label="📈 Predicted Fraud Rate",
                value=f"{predicted_rate:.2f}%",
                delta=f"{delta:+.2f}% vs actual",
                help="Percentage of transactions predicted as fraud by the model"
            )

        st.divider()

# ============================================================================
# SECTION 3: Performance Comparison Across Windows
# ============================================================================
if len(eval_rolling) > 1:
    st.header("📊 Performance Comparison Across Windows")
    st.caption("Compare model performance across different window sizes to detect drift")

    # Prepare data for comparison
    comparison_data = []
    for window_size in sorted(eval_rolling.keys()):
        data = eval_rolling[window_size]
        if data.get('count', 0) > 0:
            comparison_data.append({
                'Window Size': f"{window_size:,}",
                'Window': window_size,
                'PR-AUC': data.get('pr_auc', 0),
                'Precision': data.get('precision', 0),
                'Recall': data.get('recall', 0),
                'F1': data.get('f1', 0),
                'Actual Fraud %': data.get('actual_fraud_rate', 0) * 100,
                'Predicted Fraud %': data.get('predicted_fraud_rate', 0) * 100
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        # Metrics comparison chart
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Model Performance Metrics")
            fig_metrics = go.Figure()

            fig_metrics.add_trace(go.Bar(
                name='PR-AUC',
                x=df_comparison['Window Size'],
                y=df_comparison['PR-AUC'],
                marker_color='#3498db'
            ))
            fig_metrics.add_trace(go.Bar(
                name='Precision',
                x=df_comparison['Window Size'],
                y=df_comparison['Precision'],
                marker_color='#2ecc71'
            ))
            fig_metrics.add_trace(go.Bar(
                name='Recall',
                x=df_comparison['Window Size'],
                y=df_comparison['Recall'],
                marker_color='#e74c3c'
            ))
            fig_metrics.add_trace(go.Bar(
                name='F1',
                x=df_comparison['Window Size'],
                y=df_comparison['F1'],
                marker_color='#f39c12'
            ))

            fig_metrics.update_layout(
                barmode='group',
                xaxis_title="Window Size",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        with col_chart2:
            st.subheader("Fraud Rate Comparison")
            fig_fraud = go.Figure()

            fig_fraud.add_trace(go.Scatter(
                name='Actual Fraud Rate',
                x=df_comparison['Window Size'],
                y=df_comparison['Actual Fraud %'],
                mode='lines+markers',
                marker=dict(size=10, color='#e74c3c'),
                line=dict(width=3, color='#e74c3c')
            ))
            fig_fraud.add_trace(go.Scatter(
                name='Predicted Fraud Rate',
                x=df_comparison['Window Size'],
                y=df_comparison['Predicted Fraud %'],
                mode='lines+markers',
                marker=dict(size=10, color='#3498db'),
                line=dict(width=3, color='#3498db')
            ))

            fig_fraud.update_layout(
                xaxis_title="Window Size",
                yaxis_title="Fraud Rate (%)",
                height=400
            )
            st.plotly_chart(fig_fraud, use_container_width=True)

        # Detailed comparison table
        with st.expander("📋 Detailed Metrics Table"):
            st.dataframe(
                df_comparison.style.format({
                    'PR-AUC': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1': '{:.3f}',
                    'Actual Fraud %': '{:.2f}%',
                    'Predicted Fraud %': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

        # Drift detection alert
        if len(comparison_data) >= 2:
            smallest_window = comparison_data[0]  # Most recent
            largest_window = comparison_data[-1]  # Most historical

            pr_auc_diff = smallest_window['PR-AUC'] - largest_window['PR-AUC']
            fraud_rate_diff = smallest_window['Actual Fraud %'] - largest_window['Actual Fraud %']

            if abs(pr_auc_diff) > 0.05 or abs(fraud_rate_diff) > 2.0:
                st.warning(f"""
                ⚠️ **Potential Drift Detected**
                - PR-AUC change: {pr_auc_diff:+.3f} (smallest vs largest window)
                - Fraud rate change: {fraud_rate_diff:+.2f}% (smallest vs largest window)

                Consider retraining the model if performance continues to degrade.
                """)

st.divider()

# ============================================================================
# Footer
# ============================================================================
st.caption(f"🔄 Auto-refreshing every {REFRESH_INTERVAL} seconds | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"💡 Threshold: {fraud_threshold:.2f} | Monitoring {len(window_sizes)} rolling window(s)")
