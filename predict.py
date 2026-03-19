"""
Single Prediction Script
Predicts if a given network traffic sample is attack or normal
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load('model/nids_model.pkl')
le = joblib.load('model/label_encoder.pkl')
feature_names = joblib.load('model/feature_names.pkl')

def predict_single(values):
    df = pd.DataFrame([values], columns=feature_names)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    label = le.inverse_transform([pred])[0]
    confidence = max(proba) * 100
    return label, confidence

if __name__ == "__main__":
    print("=== Network Traffic Predictor ===\n")
    print(f"Required features: {feature_names}\n")
    
    # Test with sample values
    sample_attack = {f: 1000.0 for f in feature_names}
    sample_normal = {f: 5.0 for f in feature_names}
    
    label, conf = predict_single(sample_attack)
    print(f"Sample 1 → {label.upper()} (Confidence: {conf:.2f}%)")
    
    label, conf = predict_single(sample_normal)
    print(f"Sample 2 → {label.upper()} (Confidence: {conf:.2f}%)")