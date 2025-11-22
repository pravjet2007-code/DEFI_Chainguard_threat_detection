import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest # <--- CHANGED
import joblib
import os
import hashlib
# FIX: Absolute import
from src.data_loader import load_data

# --- Configuration ---
CONTAMINATION_RATE = 0.01  # Estimated percentage of anomalies
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
ADDRESS_COLS = ['from_address', 'to_address']
SEQUENCE_LENGTH = 1 # Not used, but kept for future reference

# --- HASHING FUNCTION ---
def secure_hash(address, salt="chainguard_secret_key_2025"):
    if pd.isna(address): return None
    return hashlib.sha256((str(address) + salt).encode()).hexdigest()

def train_and_save_model(df):
    
    # 1. PREPARE DATA (Sort by time is mandatory for time features)
    print("Processing data for Isolation Forest...")
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
    df = df.sort_values('block_timestamp').reset_index(drop=True)
    
    # 2. PRIVACY HASHING
    for col in ADDRESS_COLS:
        df[f'{col}_hashed'] = df[col].apply(secure_hash)

    # 3. ADVANCED FEATURE ENGINEERING (9 Features)
    df['value_eth'] = df['value'].astype(float) / 10**18
    df['tx_fee'] = df['gas'].astype(float) * df['gas_price'].astype(float)
    df['hour'] = df['block_timestamp'].dt.hour
    
    for col in ['from_address_hashed', 'to_address_hashed']:
        df[f'{col}_tx_count'] = df.groupby(col)['block_timestamp'].transform('count')
    
    # Z-Score of Value
    history = df.groupby('from_address_hashed')['value_eth'].agg(['mean', 'std'])
    history.columns = ['sender_mean', 'sender_std']
    df = df.merge(history, left_on='from_address_hashed', right_index=True, how='left')
    df['value_z_score'] = ((df['value_eth'] - df['sender_mean']) / df['sender_std'])
    # FIX: Handle Inf/NaN values created by division by zero
    df['value_z_score'] = np.nan_to_num(df['value_z_score'], nan=0.0, posinf=0.0, neginf=0.0)

    # Time Since Last Transaction (Feature 9)
    df['time_since_last_sender_tx'] = df.groupby('from_address_hashed')['block_timestamp'].diff().dt.total_seconds().fillna(0)

    # 4. SELECT FINAL 9 FEATURES (2D ARRAY)
    features_to_model = [
        'value_eth', 'tx_fee', 'hour', 
        'from_address_hashed_tx_count', 'to_address_hashed_tx_count', 
        'value_z_score', 'gas', 'gas_price', 'time_since_last_sender_tx'
    ]
    X = df[features_to_model].fillna(0).values

    # 5. SCALE DATA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nStarting Isolation Forest Model Training...")

    # 6. TRAINING (Deliverable 1)
    model = IsolationForest(
        n_estimators=200, 
        contamination=CONTAMINATION_RATE, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    print("Isolation Forest Training Complete!!")

    # 7. SAVE ARTIFACTS
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    # Model is saved as .pkl (Scikit-learn format)
    joblib.dump(model, os.path.join(MODEL_DIR, 'chainguard_if_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'chainguard_scaler.pkl'))
    
    print("Model saved successfully as .pkl")

if __name__ == "__main__":
    data_frame = load_data()
    if data_frame is not None:
        train_and_save_model(data_frame)