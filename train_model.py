import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

os.makedirs('model', exist_ok=True)

print("Loading datasets...")

# Load attack data
df_attack = pd.read_csv('data/DrDoS_DNS.csv')

# Standardize column names — lowercase + strip spaces
df_attack.columns = df_attack.columns.str.lower().str.strip().str.replace(' ', '_')
df_attack['label'] = 'attack'

# Load syn parquet
df_syn = pd.read_parquet('data/Syn-training.parquet')
df_syn.columns = df_syn.columns.str.lower().str.strip().str.replace(' ', '_')

print(f"\nDrDoS columns: {list(df_attack.columns)}")
print(f"\nSyn columns: {list(df_syn.columns)}")

# Get benign rows from syn
if 'label' in df_syn.columns:
    df_benign = df_syn[df_syn['label'].astype(str).str.lower() == 'benign'].copy()
    df_benign['label'] = 'normal'
else:
    df_benign = df_syn.copy()
    df_benign['label'] = 'normal'

print(f"\nAttack records: {len(df_attack)}")
print(f"Normal records: {len(df_benign)}")

# Find common columns
attack_cols = set(df_attack.columns) - {'label'}
benign_cols = set(df_benign.columns) - {'label'}
common_cols = list(attack_cols & benign_cols)

print(f"\nCommon features found: {len(common_cols)}")
print(f"Common columns: {common_cols}")

# If still no common columns — use only DrDoS and create synthetic normal data
if len(common_cols) == 0:
    print("\nNo common columns — using DrDoS only with synthetic normal data...")
    
    feature_cols = [c for c in df_attack.columns if c != 'label']
    
    # Create synthetic normal traffic (lower values = normal traffic)
    np.random.seed(42)
    n_normal = len(df_attack)
    normal_data = {}
    for col in feature_cols:
        col_mean = df_attack[col].mean()
        col_std = df_attack[col].std()
        # Normal traffic has much lower values than DDoS
        normal_data[col] = np.abs(np.random.normal(
            col_mean * 0.1, col_std * 0.1, n_normal
        ))
    
    df_benign_synth = pd.DataFrame(normal_data)
    df_benign_synth['label'] = 'normal'
    
    df = pd.concat([df_attack, df_benign_synth], ignore_index=True)
    common_cols = feature_cols

else:
    df_attack_clean = df_attack[common_cols + ['label']]
    df_benign_clean = df_benign[common_cols + ['label']]
    df = pd.concat([df_attack_clean, df_benign_clean], ignore_index=True)

# Clean data
df = df.dropna()
df = df.replace([np.inf, -np.inf], 0)

print(f"\nTotal records for training: {len(df)}")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Encode any remaining object columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, 'model/nids_model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

print("\n✅ Model saved successfully!")