"""Streamlit frontend for churn prediction."""

import streamlit as st
import requests
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
model_choice = st.sidebar.selectbox("Model", ["xgboost", "baseline"])

# Check API
try:
    resp = requests.get(f"{API_URL}/health", timeout=2)
    st.sidebar.success("✅ API Connected" if resp.status_code == 200 else "⚠️ API Error")
except:
    st.sidebar.error("⚠️ Cannot connect to API")
    st.stop()

st.title("📊 Customer Churn Prediction")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        acquisition_channel = st.selectbox("Acquisition Channel", 
            ["phone", "online", "outgoing call - promo", "pos", "mail", "outgoing call", "other"])
        fiber_or_adsl = st.selectbox("Connection Type", ["fiber", "adsl"])
        has_retention = st.checkbox("Has Retention Offer")
        offer = st.selectbox("Offer", [
            "#11:Freebox Revolution with TV 3999eur",
            "#8:Freebox Revolution 2999eur"
        ])
    with col2:
        sub_offer = st.selectbox("Sub-Offer", [
            "11.4:Freebox Revolution with TV 3999eur",
            "8.2:Freebox Revolution 2999eur",
            "8.57:Freebox Revolution 2999eur | Promo : 999eur for 1 year"
        ])
        recruit_year_month = st.text_input("Recruitment Date (YYYY-MM)", value="2016-01")
        total_bill = st.number_input("Total Bill (€)", min_value=0.0, value=1500.0)
        cancel_year_month = st.text_input("Cancellation Date (YYYY-MM) - Leave empty if active", value="")
    
    if st.button("🔮 Predict", type="primary"):
        try:
            resp = requests.post(
                f"{API_URL}/predict?model={model_choice}",
                json={
                    "acquisition_channel": acquisition_channel,
                    "fiber_or_adsl": fiber_or_adsl,
                    "has_retention": has_retention,
                    "offer": offer,
                    "sub_offer": sub_offer,
                    "recruit_year_month": recruit_year_month,
                    "total_bill": total_bill,
                    "cancel_year_month": cancel_year_month or None,
                    "duration_month": None
                },
                timeout=10
            )
            if resp.status_code == 200:
                result = resp.json()
                st.success("✅ Prediction Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
                with col2:
                    st.metric("Prediction", "⚠️ High Risk" if result['churn_prediction'] == 1 else "✅ Low Risk")
                st.progress(result['churn_probability'])
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        if st.button("🔮 Predict Batch", type="primary"):
            try:
                requests_data = [{
                    "acquisition_channel": str(row['acquisition_channel']),
                    "fiber_or_adsl": str(row['fiber_or_adsl']),
                    "has_retention": bool(row['has_retention']),
                    "offer": str(row['offer']),
                    "sub_offer": str(row['sub_offer']),
                    "recruit_year_month": str(row['recruit_year_month']),
                    "total_bill": float(row['total_bill']),
                    "cancel_year_month": str(row.get('cancel_year_month', '')) or None,
                    "duration_month": None
                } for _, row in df.iterrows()]
                
                resp = requests.post(f"{API_URL}/predict/batch?model={model_choice}", json=requests_data, timeout=30)
                if resp.status_code == 200:
                    results = resp.json()['predictions']
                    df['churn_probability'] = [r['churn_probability'] for r in results]
                    df['churn_prediction'] = [r['churn_prediction'] for r in results]
                    st.success(f"✅ Predictions for {len(df)} customers")
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), 
                                     file_name="predictions.csv", mime="text/csv")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

