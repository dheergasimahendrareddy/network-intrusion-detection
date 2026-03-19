import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model/nids_model.pkl')
le = joblib.load('model/label_encoder.pkl')
feature_names = joblib.load('model/feature_names.pkl')

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("Detects malicious network traffic using Machine Learning (Random Forest)")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Info")
    st.info(f"Features: {len(feature_names)}")
    st.info(f"Classes: {list(le.classes_)}")
    st.info("Model: Random Forest Classifier")

with col2:
    st.subheader("📁 Dataset Info")
    st.info("Attack Data: CIC-DDoS2019 DrDoS_DNS")
    st.info("Normal Data: CIC-DDoS2019 Syn-training")
    st.info("Combined: ~34,000+ records")

st.divider()
st.subheader("🔍 Test with Your Own Values")
st.markdown("Enter network traffic values below to predict if it's an attack or normal:")

input_data = {}
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            f"{feature}", value=0.0, format="%.4f"
        )

if st.button("🔍 Predict Traffic", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    label = le.inverse_transform([prediction])[0]

    if label == 'attack':
        st.error(f"🚨 ATTACK DETECTED! Confidence: {max(proba)*100:.2f}%")
    else:
        st.success(f"✅ NORMAL Traffic. Confidence: {max(proba)*100:.2f}%")

st.divider()
st.subheader("📂 Batch Prediction — Upload CSV")
uploaded = st.file_uploader("Upload a CSV file of network traffic", type=['csv'])

if uploaded:
    df_upload = pd.read_csv(uploaded)
    missing = [f for f in feature_names if f not in df_upload.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
    else:
        preds = model.predict(df_upload[feature_names])
        labels = le.inverse_transform(preds)
        df_upload['Prediction'] = labels
        attack_count = sum(1 for l in labels if l == 'attack')
        normal_count = len(labels) - attack_count
        st.success(f"✅ Normal: {normal_count} | 🚨 Attacks: {attack_count}")
        st.dataframe(df_upload)

st.divider()
st.caption("Built with Python | Scikit-learn | Streamlit | CIC-DDoS2019 Dataset")