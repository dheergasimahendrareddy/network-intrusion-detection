"""
Firewall Integration Module
============================
Demonstrates how NIDS model integrates with Linux 
firewall (iptables) to auto-block suspicious IPs.
"""

import joblib
import numpy as np
from datetime import datetime

model = joblib.load('model/nids_model.pkl')
le = joblib.load('model/label_encoder.pkl')
feature_names = joblib.load('model/feature_names.pkl')

def predict_traffic(features):
    prediction = model.predict([features])[0]
    return le.inverse_transform([prediction])[0]

def block_ip(ip_address, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    command = f"iptables -A INPUT -s {ip_address} -j DROP"
    log = f"[{timestamp}] BLOCKED: {ip_address} | Reason: {reason}\n"
    with open('blocked_ips.log', 'a') as f:
        f.write(log)
    print(f"🚫 BLOCKED IP: {ip_address} at {timestamp}")
    print(f"   Linux Command: {command}")

def monitor(features, source_ip):
    result = predict_traffic(features)
    if result == 'attack':
        print(f"⚠️  ATTACK detected from {source_ip}!")
        block_ip(source_ip, "ML Model detected DDoS attack")
        return {"status": "BLOCKED", "ip": source_ip}
    else:
        print(f"✅ Normal traffic from {source_ip}")
        return {"status": "ALLOWED", "ip": source_ip}

if __name__ == "__main__":
    print("=== NIDS Firewall Integration Demo ===\n")
    
    # Simulate attack traffic
    attack = [float(1000)] * len(feature_names)
    print("Test 1 — Attack traffic simulation:")
    monitor(attack, "192.168.1.105")
    
    # Simulate normal traffic
    normal = [float(10)] * len(feature_names)
    print("\nTest 2 — Normal traffic simulation:")
    monitor(normal, "192.168.1.200")
    
    print("\n✅ Check blocked_ips.log for blocked IP records")
    