"""
Streamlit frontend for churn prediction.

Interactive web interface for making churn predictions.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# API endpoint (default to localhost)
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of the FastAPI backend"
)

# Title
st.title("📊 Customer Churn Prediction")
st.markdown("Predict customer churn probability using machine learning models")

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["xgboost", "baseline"],
    help="Choose the ML model for prediction"
)

# Check API connection
@st.cache_data(ttl=60)
def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None


api_healthy, health_data = check_api_health()

if not api_healthy:
    st.error(f"⚠️ Cannot connect to API at {API_URL}")
    st.info("Please make sure the FastAPI server is running:\n```bash\nuvicorn churn_prediction.app.api:app --reload\n```")
    st.stop()

st.sidebar.success(f"✅ API Connected")
if health_data:
    st.sidebar.json(health_data)

# Main interface
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        acquisition_channel = st.selectbox(
            "Acquisition Channel",
            ["phone", "online", "outgoing call - promo", "pos", "mail", "outgoing call", "other"]
        )
        
        fiber_or_adsl = st.selectbox(
            "Connection Type",
            ["fiber", "adsl"]
        )
        
        has_retention = st.checkbox("Has Retention Offer", value=False)
        
        offer = st.selectbox(
            "Offer",
            [
                "#11:Freebox Revolution with TV 3999eur",
                "#8:Freebox Revolution 2999eur"
            ]
        )
    
    with col2:
        sub_offer = st.selectbox(
            "Sub-Offer",
            [
                "11.4:Freebox Revolution with TV 3999eur",
                "8.2:Freebox Revolution 2999eur",
                "8.57:Freebox Revolution 2999eur | Promo : 999eur for 1 year"
            ]
        )
        
        recruit_year_month = st.text_input(
            "Recruitment Date (YYYY-MM)",
            value="2016-01",
            help="Format: YYYY-MM"
        )
        
        total_bill = st.number_input(
            "Total Bill (€)",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )
        
        cancel_year_month = st.text_input(
            "Cancellation Date (YYYY-MM) - Leave empty if active",
            value="",
            help="Leave empty for active customers"
        )
    
    # Predict button
    if st.button("🔮 Predict Churn", type="primary"):
        with st.spinner("Making prediction..."):
            try:
                # Prepare request
                request_data = {
                    "acquisition_channel": acquisition_channel,
                    "fiber_or_adsl": fiber_or_adsl,
                    "has_retention": has_retention,
                    "offer": offer,
                    "sub_offer": sub_offer,
                    "recruit_year_month": recruit_year_month,
                    "total_bill": total_bill,
                    "cancel_year_month": cancel_year_month if cancel_year_month else None,
                    "duration_month": None
                }
                
                # Make API call
                response = requests.post(
                    f"{API_URL}/predict?model={model_choice}",
                    json=request_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Churn Probability",
                            f"{result['churn_probability']:.1%}"
                        )
                    
                    with col2:
                        prediction_label = "⚠️ High Risk" if result['churn_prediction'] == 1 else "✅ Low Risk"
                        st.metric(
                            "Prediction",
                            prediction_label
                        )
                    
                    with col3:
                        st.metric(
                            "Model Used",
                            result['model_used'].upper()
                        )
                    
                    # Probability bar
                    st.progress(result['churn_probability'])
                    
                    # Interpretation
                    if result['churn_probability'] > 0.7:
                        st.warning("🔴 High churn risk detected. Consider retention strategies.")
                    elif result['churn_probability'] > 0.4:
                        st.info("🟡 Moderate churn risk. Monitor customer engagement.")
                    else:
                        st.success("🟢 Low churn risk. Customer appears stable.")
                
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with customer data for batch predictions")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV file should contain columns: acquisition_channel, fiber_or_adsl, has_retention, offer, sub_offer, recruit_year_month, total_bill, cancel_year_month (optional)"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            # Required columns
            required_cols = [
                'acquisition_channel', 'fiber_or_adsl', 'has_retention',
                'offer', 'sub_offer', 'recruit_year_month', 'total_bill'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🔮 Predict Batch", type="primary"):
                    with st.spinner(f"Predicting churn for {len(df)} customers..."):
                        try:
                            # Prepare batch request
                            requests_data = []
                            for _, row in df.iterrows():
                                requests_data.append({
                                    "acquisition_channel": str(row['acquisition_channel']),
                                    "fiber_or_adsl": str(row['fiber_or_adsl']),
                                    "has_retention": bool(row['has_retention']),
                                    "offer": str(row['offer']),
                                    "sub_offer": str(row['sub_offer']),
                                    "recruit_year_month": str(row['recruit_year_month']),
                                    "total_bill": float(row['total_bill']),
                                    "cancel_year_month": str(row.get('cancel_year_month', '')) or None,
                                    "duration_month": None
                                })
                            
                            # Make API call
                            response = requests.post(
                                f"{API_URL}/predict/batch?model={model_choice}",
                                json=requests_data,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                results = response.json()['predictions']
                                
                                # Add predictions to dataframe
                                df_results = df.copy()
                                df_results['churn_probability'] = [r['churn_probability'] for r in results]
                                df_results['churn_prediction'] = [r['churn_prediction'] for r in results]
                                
                                st.success(f"✅ Predictions complete for {len(df_results)} customers!")
                                
                                # Display results
                                st.dataframe(df_results, use_container_width=True)
                                
                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("High Risk Customers", f"{(df_results['churn_prediction'] == 1).sum()}")
                                with col2:
                                    st.metric("Average Churn Probability", f"{df_results['churn_probability'].mean():.1%}")
                                with col3:
                                    st.metric("Low Risk Customers", f"{(df_results['churn_prediction'] == 0).sum()}")
                                
                                # Download results
                                csv = df_results.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=csv,
                                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            else:
                                st.error(f"❌ Batch prediction failed: {response.text}")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Churn Prediction System** | Built with FastAPI & Streamlit")

