# 🛡️ AI-Powered Network Intrusion Detection System

Detects malicious DDoS network traffic using Machine Learning
with **97.61% accuracy** on real-world data.

## 🎯 What This Project Does
This is not just an ML model — it is a complete Security 
Integration System designed to connect with a Linux firewall 
(iptables) to automatically block suspicious IPs in real-time.

## 🛠️ Tech Stack
- Python 3.10
- Scikit-learn (Random Forest Classifier)
- Streamlit (Live Web Dashboard)
- CIC-DDoS2019 Dataset (Real world network traffic)
- Pandas, NumPy, Joblib

## 📊 Model Performance
| Metric | Value |
|---|---|
| Accuracy | **97.61%** |
| Training Records | 104,000+ |
| Model | Random Forest |
| Classes | Attack / Normal |

## ⚔️ Attack Types Detected
- DrDoS_DNS (DNS Amplification DDoS Attack)
- Normal vs Malicious traffic classification

## 🔗 System Integration Flow
```
Live Network Traffic
        ↓
   NIDS ML Model
        ↓
  Attack Detected?
   ↓          ↓
  YES          NO
   ↓           ↓
Block IP    Allow Traffic
(iptables)
```

## 📁 Project Structure
```
network-intrusion-detection/
├── data/                      # Training datasets
├── model/                     # Saved ML model
├── train_model.py             # Model training script
├── app.py                     # Streamlit web dashboard
├── firewall_integration.py    # Linux firewall integration
├── predict.py                 # Single prediction script
└── requirements.txt
```

## ⚙️ How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run web dashboard
streamlit run app.py

# Test firewall integration
python firewall_integration.py
```

## 👨‍💻 Author
mahendra reddy | B.Tech CSE | Lovely Professional University | 2026
