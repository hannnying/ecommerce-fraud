"""
Streamlit Dashboard for Fraud Detection System
Focuses on 3 core features:
1. Risky Transactions Monitoring
2. Model Performance Evaluation
3. Data Drift Detection
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.config import API_PORT, BACKEND_HOST
from streamlit_autorefresh import st_autorefresh

# ============================================================================
# Configuration
# ============================================================================
BACKEND_URL = f"http://{BACKEND_HOST}:{API_PORT}"

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Sidebar Settings
# ============================================================================
st.sidebar.header("⚙️ Settings")

# Auto-refresh
refresh_interval = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=1,
    max_value=60,
    value=5,
    step=1
)
st_autorefresh(interval=refresh_interval * 1000, limit=None)

# Model-specific thresholds
st.sidebar.subheader("🎯 Fraud Thresholds")
threshold_seen = st.sidebar.slider(
    "Seen Devices",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)
threshold_unseen = st.sidebar.slider(
    "Unseen Devices",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05
)
threshold_rules = st.sidebar.slider(
    "Rule-Based",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.05
)

# Window sizes for business metrics and drift detection
st.sidebar.subheader("📊 Analysis Settings")
window_sizes = st.sidebar.multiselect(
    "Window Sizes",
    options=[1000, 5000, 10000, 20000],
    default=[5000],
    help="Number of recent transactions to analyze"
)

# Drift threshold
drift_threshold = st.sidebar.slider(
    "Drift Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Jensen-Shannon divergence threshold"
)

# ============================================================================
# Helper Functions
# ============================================================================
def fetch_api(endpoint, params=None, default=None):
    """Fetch data from backend API with error handling."""
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return default
    except Exception as e:
        st.sidebar.error(f"API Error: {endpoint}")
        return default


def render_metric_row(metrics, col_configs):
    """Render a row of metrics using columns."""
    cols = st.columns(len(col_configs))
    for col, (label, key, format_func) in zip(cols, col_configs):
        with col:
            value = metrics.get(key, 0)
            st.metric(label=label, value=format_func(value))


def highlight_rows_by_column(df, column, color_map):
    """Apply row highlighting based on column value."""
    def apply_color(row):
        color = color_map.get(row[column], '#ffffff')
        return [f'background-color: {color}'] * len(row)
    return apply_color


# ============================================================================
# Main Title
# ============================================================================
st.title("🛡️ Fraud Detection Monitor")

# ============================================================================
# Create Tabs: Risky Transactions, Business Impact, Data Drift
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    "🚨 Risky Transactions",
    "💰 Business Impact",
    "🔍 Data Drift"
])

# ============================================================================
# TAB 1: RISKY TRANSACTIONS
# ============================================================================
with tab1:
    st.header("🚨 High-Risk Transactions")
    st.caption("Transactions flagged by model-specific thresholds")

    # Fetch risky transactions from API (British spelling: analyse)
    risky_txns = fetch_api(
        "/transactions/analyse",
        params={
            "threshold_seen": threshold_seen,
            "threshold_unseen": threshold_unseen,
            "threshold_rules": threshold_rules,
            "limit": 100
        }
    )

    if risky_txns and risky_txns.get("transactions"):
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 Total Flagged", risky_txns.get("total_count", 0))
        with col2:
            st.metric("🆕 Unseen Devices", risky_txns.get("unseen_count", 0))
        with col3:
            st.metric("👁️ Seen Devices", risky_txns.get("seen_count", 0))
        with col4:
            st.metric("📋 Rule-Based", risky_txns.get("rules_count", 0))

        st.divider()

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            model_filter = st.multiselect(
                "Filter by Model",
                options=["seen_devices", "unseen_devices", "rule_based"],
                default=["seen_devices", "unseen_devices", "rule_based"]
            )
        with col_f2:
            sort_by = st.selectbox(
                "Sort by",
                options=["fraud_proba_desc", "fraud_proba_asc", "purchase_time_desc"],
                format_func=lambda x: {
                    "fraud_proba_desc": "Fraud Prob (High → Low)",
                    "fraud_proba_asc": "Fraud Prob (Low → High)",
                    "purchase_time_desc": "Most Recent First"
                }[x]
            )

        # Process and display transactions
        df_txns = pd.DataFrame(risky_txns["transactions"])

        # Rename risk_score to fraud_proba for consistency
        if "risk_score" in df_txns.columns:
            df_txns = df_txns.rename(columns={"risk_score": "fraud_proba"})

        # Apply filters
        if model_filter:
            df_txns = df_txns[df_txns["model_used"].isin(model_filter)]

        # Apply sorting
        sort_map = {
            "fraud_proba_desc": ("fraud_proba", False),
            "fraud_proba_asc": ("fraud_proba", True),
            "purchase_time_desc": ("purchase_time", False)
        }
        sort_col, ascending = sort_map[sort_by]
        df_txns = df_txns.sort_values(sort_col, ascending=ascending)

        if len(df_txns) > 0:
            st.subheader(f"📊 {len(df_txns)} Transaction(s)")

            # Select display columns
            display_cols = [
                "transaction_id", "device_id", "fraud_proba", "model_used",
                "purchase_time", "device_txn_idx"
            ]
            optional_cols = [
                "device_age_hours", "device_txn_velocity_24h",
                "purchase_hour", "source", "browser"
            ]
            display_cols.extend([c for c in optional_cols if c in df_txns.columns])
            display_cols = [c for c in display_cols if c in df_txns.columns]

            # Color-code by model
            color_map = {
                "rule_based": "#ffe6e6",
                "unseen_devices": "#fff4e6",
                "seen_devices": "#e6f3ff"
            }

            st.dataframe(
                df_txns[display_cols].style.apply(
                    highlight_rows_by_column(df_txns, "model_used", color_map),
                    axis=1
                ).format({
                    "fraud_proba": "{:.4f}",
                    "device_age_hours": "{:.2f}" if "device_age_hours" in df_txns else None,
                    "purchase_time": lambda x: str(x)[:19] if pd.notna(x) else ""
                }),
                use_container_width=True,
                height=400
            )

            st.caption("🔴 Red: Rule-Based | 🟠 Orange: Unseen Devices | 🔵 Blue: Seen Devices")

            # Visualizations
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.subheader("Distribution by Model")
                model_counts = df_txns["model_used"].value_counts()
                fig_pie = px.pie(
                    values=model_counts.values,
                    names=model_counts.index,
                    hole=0.3,
                    color_discrete_sequence=['#3498db', '#f39c12', '#e74c3c']
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_v2:
                st.subheader("Fraud Probability Distribution")
                fig_hist = px.histogram(
                    df_txns,
                    x="fraud_proba",
                    color="model_used",
                    nbins=20,
                    opacity=0.7,
                    barmode='overlay'
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No transactions match the selected filters.")

    elif risky_txns and risky_txns.get("message"):
        st.info(f"ℹ️ {risky_txns['message']}")
    else:
        st.warning("⏳ No high-risk transactions found or data not available yet.")


# ============================================================================
# TAB 2: BUSINESS IMPACT
# ============================================================================
with tab2:
    st.header("💰 Business Impact Evaluation")
    st.caption("Financial impact of fraud detection based on transaction values")

    # Get business metrics using smallest window size
    business_window = min(window_sizes) if window_sizes else 5000
    business_metrics = fetch_api(
        "/evaluate/business",
        params={
            "window_size": business_window,
            "threshold_seen": threshold_seen,
            "threshold_unseen": threshold_unseen,
            "threshold_rules": threshold_rules
        }
    )

    if business_metrics and business_metrics.get("total_transaction_value", 0) > 0:
        # Extract values
        total_value = business_metrics.get("total_transaction_value", 0)
        fraud_caught = business_metrics.get("fraud_caught_value", 0)
        genuine = business_metrics.get("genuine_value", 0)
        missed_fraud = business_metrics.get("missed_fraud_value", 0)

        # Display main business metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💳 Total Transaction Value",
                f"${total_value:,.2f}",
                help="Total dollar value of all processed transactions"
            )

        with col2:
            fraud_caught_pct = (fraud_caught / total_value * 100) if total_value > 0 else 0
            st.metric(
                "🛡️ Fraud Caught Value",
                f"${fraud_caught:,.2f}",
                delta=f"{fraud_caught_pct:.2f}% of total",
                help="Dollar value of fraud successfully detected"
            )

        with col3:
            genuine_pct = (genuine / total_value * 100) if total_value > 0 else 0
            st.metric(
                "✅ Genuine Value",
                f"${genuine:,.2f}",
                delta=f"{genuine_pct:.2f}% of total",
                help="Dollar value of legitimate transactions"
            )

        with col4:
            missed_fraud_pct = (missed_fraud / total_value * 100) if total_value > 0 else 0
            st.metric(
                "⚠️ Missed Fraud Value",
                f"${missed_fraud:,.2f}",
                delta=f"{missed_fraud_pct:.2f}% of total",
                delta_color="inverse",
                help="Dollar value of fraud not detected (losses)"
            )

        st.divider()

        # System effectiveness metrics
        st.subheader("📊 System Effectiveness")

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            # Fraud detection rate
            total_fraud_value = fraud_caught + missed_fraud
            fraud_detection_rate = (fraud_caught / total_fraud_value * 100) if total_fraud_value > 0 else 0
            st.metric(
                "🎯 Fraud Detection Rate",
                f"{fraud_detection_rate:.2f}%",
                help="Percentage of fraud value caught (Fraud Caught / Total Fraud)"
            )

        with col_e2:
            # Value protection rate
            protection_rate = ((fraud_caught + genuine) / total_value * 100) if total_value > 0 else 0
            st.metric(
                "🛡️ Value Protection Rate",
                f"{protection_rate:.2f}%",
                help="Percentage correctly classified (Fraud Caught + Genuine) / Total"
            )

        st.divider()

        # Visualizations
        st.subheader("💹 Transaction Value Breakdown")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            # Pie chart
            fig_pie = px.pie(
                values=[fraud_caught, genuine, missed_fraud],
                names=["Fraud Caught", "Genuine", "Missed Fraud"],
                title="Value Distribution",
                color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c'],
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_v2:
            # Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["Fraud Caught", "Genuine", "Missed Fraud"],
                y=[fraud_caught, genuine, missed_fraud],
                marker_color=['#2ecc71', '#3498db', '#e74c3c'],
                text=[f"${fraud_caught:,.0f}", f"${genuine:,.0f}", f"${missed_fraud:,.0f}"],
                textposition='outside'
            ))
            fig_bar.update_layout(
                title="Absolute Dollar Values",
                yaxis_title="Transaction Value ($)",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Alerts for poor performance
        if missed_fraud_pct > 10:
            st.error(f"""
            🚨 **High Missed Fraud Alert**

            Missed fraud represents {missed_fraud_pct:.2f}% of total value (${missed_fraud:,.2f}).

            **Recommendations:**
            - Review and lower fraud detection thresholds
            - Check Data Drift tab for distribution changes
            - Consider model retraining
            """)

        if fraud_detection_rate < 80 and total_fraud_value > 0:
            st.warning(f"""
            ⚠️ **Moderate Fraud Detection Performance**

            System catching {fraud_detection_rate:.2f}% of fraud value.
            - Adjust thresholds in sidebar
            - Check for data drift
            - Review recent false negatives
            """)

        # Info panel
        with st.expander("ℹ️ About Business Metrics"):
            st.markdown(f"""
            **Data Source:** Last {business_window:,} transactions with ground-truth labels

            **Metric Definitions:**
            - **Fraud Caught**: Transactions correctly identified as fraud
            - **Genuine**: Legitimate transactions correctly processed
            - **Missed Fraud**: Fraudulent transactions not detected (losses)

            **Thresholds:** Seen: {threshold_seen:.2f} | Unseen: {threshold_unseen:.2f} | Rules: {threshold_rules:.2f}
            """)

    else:
        st.warning("""
        ⏳ **Waiting for labeled transaction data...**

        Business metrics require transactions with ground-truth labels.

        **Check:**
        - Producer streaming transactions (`python api/producer.py`)
        - Consumer processing them (`python api/consumer.py`)
        - Labels sent to Redis LABELS_STREAM
        """)


# ============================================================================
# TAB 3: DATA DRIFT
# ============================================================================
with tab3:
    st.header("🔍 Data Drift Detection")
    st.caption("Monitors feature distribution changes vs. training data")

    # Fetch drift data
    drift_window = min(window_sizes) if window_sizes else 5000
    drift_data = fetch_api(
        "/drift/check",
        params={"window_size": drift_window, "threshold": drift_threshold}
    )

    if drift_data and "drift_results" in drift_data and drift_data["drift_results"]:
        num_drifted = drift_data.get("num_features_drifted", 0)
        total_features = drift_data.get("total_features_monitored", 0)
        should_retrain = drift_data.get("should_retrain", False)

        # Alert status
        if should_retrain:
            st.error(f"⚠️ **Significant drift detected!** {num_drifted}/{total_features} features drifted. Retraining recommended.")
        elif num_drifted > 0:
            st.warning(f"⚡ Moderate drift in {num_drifted} feature(s). Continue monitoring.")
        else:
            st.success(f"✅ No significant drift across {total_features} features.")

        # Summary metrics
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Features Monitored", total_features)
        with col_d2:
            pct = (num_drifted/total_features*100) if total_features > 0 else 0
            st.metric("Features Drifted", num_drifted, delta=f"{pct:.1f}%", delta_color="inverse")
        with col_d3:
            st.metric("Window Size", f"{drift_data.get('window_size', 0):,}")

        st.divider()

        # Drift results table
        st.subheader("Drift Scores by Feature")

        drift_results = drift_data["drift_results"]
        drift_table = []
        for feature, result in sorted(drift_results.items(), key=lambda x: x[1]["drift_score"], reverse=True):
            drift_table.append({
                "Feature": feature,
                "Type": result["type"].capitalize(),
                "Drift Score": result["drift_score"],
                "Status": "🔴 Drifted" if result["drifted"] else "🟢 OK",
                "_drifted": result["drifted"]
            })

        df_drift = pd.DataFrame(drift_table)

        # Color-code table
        def highlight_drift(row):
            color = '#ffcccc' if row["_drifted"] else '#ccffcc'
            return [f'background-color: {color}'] * len(row)

        st.dataframe(
            df_drift.drop(columns=["_drifted"]).style.apply(highlight_drift, axis=1).format({
                "Drift Score": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Drift visualization
        st.subheader("Drift Score Visualization")

        fig_drift = go.Figure()
        colors = ['#e74c3c' if row["_drifted"] else '#2ecc71' for _, row in df_drift.iterrows()]

        fig_drift.add_trace(go.Bar(
            x=df_drift["Feature"],
            y=df_drift["Drift Score"],
            marker_color=colors,
            text=df_drift["Drift Score"].round(4),
            textposition='outside'
        ))

        # Threshold line
        fig_drift.add_hline(
            y=drift_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold ({drift_threshold})",
            annotation_position="right"
        )

        fig_drift.update_layout(
            xaxis_title="Feature",
            yaxis_title="Jensen-Shannon Divergence",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_drift, use_container_width=True)

        # Info about drift metric
        with st.expander("ℹ️ About Drift Detection"):
            st.markdown(f"""
            **Jensen-Shannon Divergence (JSD)** measures distribution similarity:
            - **0.0** = Identical (no drift)
            - **1.0** = Completely different (maximum drift)

            **Interpretation:**
            - JSD < 0.1: Minimal drift
            - 0.1 ≤ JSD < 0.2: Moderate drift (monitor)
            - JSD ≥ 0.2: Significant drift (retrain)

            **Current Settings:**
            - Threshold: {drift_threshold}
            - Window: {drift_window:,} transactions
            - Baseline: Training data (run `{drift_data.get('run_id', 'unknown')}`)
            """)

    elif drift_data and "message" in drift_data:
        st.info(f"ℹ️ {drift_data['message']}")
    else:
        st.warning("⏳ Drift monitoring not available. Ensure model is trained and transactions are streaming.")


# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption(f"🔄 Auto-refresh: {refresh_interval}s | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"💡 Active Thresholds → Seen Devices: {threshold_seen:.2f} | Unseen Devices: {threshold_unseen:.2f} | Rule-Based: {threshold_rules:.2f}")
