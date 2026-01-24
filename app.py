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
    page_title="E-commerce Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ E-commerce Fraud Detection Dashboard")

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

# Transaction display settings
k = st.sidebar.slider(
    "Recent transactions to display",
    min_value=10,
    max_value=100,
    value=50,
    step=10
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

high_risk_threshold = st.sidebar.slider(
    "High-risk alert threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Transactions above this threshold trigger high-risk alerts"
)

# Filter settings
st.sidebar.subheader("🔍 Filters")
show_fraud_only = st.sidebar.checkbox("Show fraud only", value=False)
show_legitimate_only = st.sidebar.checkbox("Show legitimate only", value=False)

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

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# ============================================================================
# Fetch Data from Backend
# ============================================================================
# Existing endpoint
recent_data = fetch_data("/results/recent", params={"k": k}, default={"results": []})
recent_transactions = recent_data.get("results", [])

# New endpoints (with graceful fallback)
summary_stats = fetch_data("/stats/summary", default=None)
hourly_stats = fetch_data("/stats/hourly", params={"hours": 24}, default=None)
high_risk_data = fetch_data("/results/high-risk", params={"threshold": high_risk_threshold, "k": 20}, default=None)
feature_stats = fetch_data("/stats/features", default=None)

# Existing endpoint
eval_metrics = fetch_data("/evaluate", params={"threshold": fraud_threshold}, default=None)

# ============================================================================
# PHASE 1: Statistics Summary Cards
# ============================================================================
st.header("📊 Real-Time Statistics")

# Calculate stats from recent transactions if /stats/summary not available
if summary_stats is None and recent_transactions:
    df_recent = pd.DataFrame(recent_transactions)
    df_recent['fraud_probability'] = df_recent['fraud_probability'].apply(safe_float)
    df_recent['predicted_class'] = df_recent['predicted_class'].apply(safe_int)

    total_txns = len(df_recent)
    fraud_count = int(df_recent['predicted_class'].sum())
    fraud_percentage = (fraud_count / total_txns * 100) if total_txns > 0 else 0
    avg_fraud_prob = df_recent['fraud_probability'].mean()
    high_risk_count = len(df_recent[df_recent['fraud_probability'] >= high_risk_threshold])
    legitimate_count = total_txns - fraud_count

    summary_stats = {
        "total_transactions": total_txns,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_percentage,
        "avg_fraud_probability": avg_fraud_prob,
        "high_risk_count": high_risk_count,
        "legitimate_count": legitimate_count
    }

if summary_stats:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📝 Total Transactions",
            value=f"{summary_stats.get('total_transactions', 0):,}"
        )

    with col2:
        fraud_count = summary_stats.get('fraud_count', 0)
        fraud_pct = summary_stats.get('fraud_percentage', 0)
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{fraud_count:,}",
            delta=f"{fraud_pct:.2f}%",
            delta_color="inverse"
        )

    with col3:
        avg_prob = summary_stats.get('avg_fraud_probability', 0)
        st.metric(
            label="📈 Avg Fraud Probability",
            value=f"{avg_prob:.3f}"
        )

    with col4:
        high_risk = summary_stats.get('high_risk_count', 0)
        st.metric(
            label="⚠️ High Risk (>{:.0%})".format(high_risk_threshold),
            value=f"{high_risk:,}",
            delta_color="inverse"
        )

st.divider()

# ============================================================================
# PHASE 1: Real-Time Fraud Rate Gauge
# ============================================================================
col_gauge, col_pie = st.columns([1, 1])

with col_gauge:
    st.subheader("🎯 Current Fraud Rate")
    if summary_stats:
        fraud_rate = summary_stats.get('fraud_percentage', 0)
        baseline_rate = 9.365  # From config

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Rate (%)"},
            delta={'reference': baseline_rate, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 20]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 10], 'color': "yellow"},
                    {'range': [10, 20], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': baseline_rate
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("Waiting for transaction data...")

# ============================================================================
# PHASE 2: Fraud vs Non-Fraud Visualization
# ============================================================================
with col_pie:
    st.subheader("📊 Fraud Distribution")
    if summary_stats:
        fraud_cnt = summary_stats.get('fraud_count', 0)
        legit_cnt = summary_stats.get('legitimate_count', 0)

        fig_pie = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[legit_cnt, fraud_cnt],
            hole=.4,
            marker=dict(colors=['#2ecc71', '#e74c3c'])
        )])
        fig_pie.update_layout(
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for transaction data...")

st.divider()

# ============================================================================
# PHASE 1: Model Performance Metrics
# ============================================================================
st.header("🎓 Model Performance")

if eval_metrics and eval_metrics.get('count', 0) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics_data = [
        ("Accuracy", eval_metrics.get('accuracy', 0), col1),
        ("Precision", eval_metrics.get('precision', 0), col2),
        ("Recall", eval_metrics.get('recall', 0), col3),
        ("F1 Score", eval_metrics.get('f1', 0), col4),
        ("PR-AUC", eval_metrics.get('pr_auc', 0), col5)
    ]

    for label, value, col in metrics_data:
        with col:
            st.metric(label=label, value=f"{value:.3f}")

    st.caption(f"📊 Evaluated on {eval_metrics.get('count', 0)} labeled transactions at threshold {fraud_threshold:.2f}")
else:
    st.info("Model evaluation metrics will appear once labeled transactions are available in Redis.")

st.divider()

# ============================================================================
# PHASE 2: High-Risk Alerts
# ============================================================================
st.header("⚠️ High-Risk Alerts")

if high_risk_data and high_risk_data.get('count', 0) > 0:
    high_risk_transactions = high_risk_data.get('results', [])
    st.warning(f"🚨 {len(high_risk_transactions)} high-risk transactions detected (fraud probability > {high_risk_threshold:.0%})")

    df_high_risk = pd.DataFrame(high_risk_transactions)
    df_high_risk['fraud_probability'] = df_high_risk['fraud_probability'].apply(safe_float)

    # Display only key columns for alerts
    alert_columns = ['transaction_id', 'fraud_probability', 'device_txn_velocity_24h',
                     'ip_switched', 'fast_purchase', 'identity_changed']
    display_cols = [col for col in alert_columns if col in df_high_risk.columns]

    st.dataframe(
        df_high_risk[display_cols].sort_values('fraud_probability', ascending=False),
        use_container_width=True,
        hide_index=True
    )
elif high_risk_data is None:
    st.info("💡 High-risk alerts endpoint not yet implemented. Implement `/results/high-risk` to enable this feature.")
else:
    st.success("✅ No high-risk transactions detected.")

st.divider()

# ============================================================================
# PHASE 2: Time-based Trend Charts
# ============================================================================
st.header("📈 Transaction Trends")

if hourly_stats and hourly_stats.get('hourly_stats'):
    df_hourly = pd.DataFrame(hourly_stats['hourly_stats'])
    df_hourly['hour'] = pd.to_datetime(df_hourly['hour'])

    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        st.subheader("Transaction Volume (Last 24 Hours)")
        fig_volume = px.line(
            df_hourly,
            x='hour',
            y='total_count',
            labels={'total_count': 'Transactions', 'hour': 'Time'},
            markers=True
        )
        fig_volume.update_traces(line_color='#3498db')
        st.plotly_chart(fig_volume, use_container_width=True)

    with col_trend2:
        st.subheader("Fraud Rate Trend (Last 24 Hours)")
        df_hourly['fraud_rate_pct'] = df_hourly['fraud_rate'] * 100
        fig_fraud_rate = px.line(
            df_hourly,
            x='hour',
            y='fraud_rate_pct',
            labels={'fraud_rate_pct': 'Fraud Rate (%)', 'hour': 'Time'},
            markers=True
        )
        fig_fraud_rate.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig_fraud_rate, use_container_width=True)
else:
    st.info("💡 Hourly trends endpoint not yet implemented. Implement `/stats/hourly` to enable this feature.")

st.divider()

# ============================================================================
# PHASE 2: Feature Analysis Charts
# ============================================================================
st.header("🔍 Feature Analysis")

if feature_stats:
    fraud_features = feature_stats.get('fraud_features', {})
    legit_features = feature_stats.get('legitimate_features', {})

    # Top risk indicators comparison
    st.subheader("Top Risk Indicators: Fraud vs Legitimate")

    risk_features = ['ip_switched', 'sex_changed', 'identity_changed', 'fast_purchase',
                     'device_txn_velocity_1h', 'device_txn_velocity_24h']

    feature_comparison = []
    for feature in risk_features:
        if feature in fraud_features and feature in legit_features:
            feature_comparison.append({
                'Feature': feature,
                'Fraud': fraud_features.get('feature', 0),
                'Legitimate': legit_features.get('feature', 0)
            })

    if feature_comparison:
        df_features = pd.DataFrame(feature_comparison)

        fig_features = go.Figure(data=[
            go.Bar(name='Fraud', x=df_features['Feature'], y=df_features['Fraud'], marker_color='#e74c3c'),
            go.Bar(name='Legitimate', x=df_features['Feature'], y=df_features['Legitimate'], marker_color='#2ecc71')
        ])
        fig_features.update_layout(
            barmode='group',
            xaxis_title="Feature",
            yaxis_title="Average Value",
            height=400
        )
        st.plotly_chart(fig_features, use_container_width=True)
    else:
        st.info("No feature comparison data available.")
else:
    st.info("💡 Feature analysis endpoint not yet implemented. Implement `/stats/features` to enable this feature.")

st.divider()

# ============================================================================
# PHASE 1: Enhanced Transaction Table with Filters
# ============================================================================
st.header("📋 Recent Transactions")

if recent_transactions:
    df = pd.DataFrame(recent_transactions)

    # Convert columns to appropriate types
    df['fraud_probability'] = df['fraud_probability'].apply(safe_float)
    df['predicted_class'] = df['predicted_class'].apply(safe_int)

    # Apply filters
    if show_fraud_only:
        df = df[df['predicted_class'] == 1]
    elif show_legitimate_only:
        df = df[df['predicted_class'] == 0]

    # Filter by fraud threshold
    threshold_filter = st.checkbox(f"Show only transactions above threshold ({fraud_threshold:.2f})", value=False)
    if threshold_filter:
        df = df[df['fraud_probability'] >= fraud_threshold]

    # Display transaction count
    st.caption(f"Showing {len(df)} transactions")

    # Color-code the dataframe
    def highlight_fraud(row):
        if safe_int(row['predicted_class']) == 1:
            return ['background-color: #ffcccc'] * len(row)  # Light red for fraud
        else:
            return ['background-color: #ccffcc'] * len(row)  # Light green for legitimate

    # Select columns to display
    display_columns = [
        'transaction_id', 'fraud_probability', 'predicted_class',
        'device_txn_velocity_24h', 'device_txn_velocity_1h',
        'ip_switched', 'fast_purchase', 'identity_changed', 'sex_changed'
    ]

    # Only show columns that exist
    available_columns = [col for col in display_columns if col in df.columns]

    # Sort by fraud probability
    df_display = df[available_columns].sort_values('fraud_probability', ascending=False)

    # Apply styling and display
    styled_df = df_display.style.apply(highlight_fraud, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Data as CSV",
        data=csv,
        file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.info("No transactions processed yet. Start the producer and consumer services.")

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption("🔄 Auto-refreshing every {} seconds | Last update: {}".format(
    REFRESH_INTERVAL,
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
))
