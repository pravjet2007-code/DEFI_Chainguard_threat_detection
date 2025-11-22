
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
from tabulate import tabulate
from src.data_loader import load_data  # Explicit package import

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
ADDRESS_COLS = ['from_address', 'to_address']
RISK_THRESHOLD = 95
WATCHLIST_SIZE = 50
WALLET_HIST_DIR = 'wallet_histories'  # per-wallet histories

def secure_hash(address, salt="chainguard_secret_key_2025"):
    if pd.isna(address): return None
    return hashlib.sha256((str(address).strip().lower() + salt).encode()).hexdigest()

def normalize_hash(h):
    if h is None: return None
    return str(h).strip().lower()

def safe_wallet_filename(w_hash_normalized):
    """Return a short, OS-safe filename for a wallet hash."""
    prefix = w_hash_normalized[:16]
    digest = hashlib.sha1(w_hash_normalized.encode()).hexdigest()
    return f"{prefix}_{digest}"

def generate_risk_report():
    print("🚀 Starting Analysis Pipeline...")

    # 1. Load data & model
    df_raw = load_data()
    if df_raw is None:
        print("❌ load_data() returned None")
        return

    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'chainguard_if_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'chainguard_scaler.pkl'))
    except FileNotFoundError as e:
        print(f"❌ Error loading models. Run train_model.py first! Error: {e}")
        return

    # 2. Feature engineering
    print("⚙️ Processing features...")
    # Coerce timestamp
    df_raw['block_timestamp'] = pd.to_datetime(df_raw['block_timestamp'], errors='coerce', utc=True)
    df_raw = df_raw.sort_values('block_timestamp').reset_index(drop=True)

    df_proc = df_raw.copy()
    for col in ADDRESS_COLS:
        # Normalize original addresses before hashing
        df_proc[col] = df_proc[col].astype(str).str.strip().str.lower()
        df_proc[f'{col}_hashed'] = df_proc[col].apply(secure_hash)

    # Normalize hashed columns (defensive)
    df_proc['from_address_hashed'] = df_proc['from_address_hashed'].apply(normalize_hash)
    df_proc['to_address_hashed'] = df_proc['to_address_hashed'].apply(normalize_hash)

    # Numeric coercions
    df_proc['value_eth'] = pd.to_numeric(df_proc['value'], errors='coerce') / (10**18)
    df_proc['gas'] = pd.to_numeric(df_proc['gas'], errors='coerce')
    df_proc['gas_price'] = pd.to_numeric(df_proc['gas_price'], errors='coerce')
    df_proc['tx_fee'] = df_proc['gas'] * df_proc['gas_price']
    df_proc['hour'] = df_proc['block_timestamp'].dt.hour

    for col in ['from_address_hashed', 'to_address_hashed']:
        df_proc[f'{col}_tx_count'] = df_proc.groupby(col)['block_timestamp'].transform('count')

    history = df_proc.groupby('from_address_hashed')['value_eth'].agg(['mean', 'std'])
    history.columns = ['sender_mean', 'sender_std']
    df_proc = df_proc.merge(history, left_on='from_address_hashed', right_index=True, how='left')

    df_proc['value_z_score'] = ((df_proc['value_eth'] - df_proc['sender_mean']) / df_proc['sender_std'])
    df_proc['value_z_score'] = np.nan_to_num(df_proc['value_z_score'], nan=0.0, posinf=0.0, neginf=0.0)

    df_proc['time_since_last_sender_tx'] = (
        df_proc.groupby('from_address_hashed')['block_timestamp']
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    # 3. Predict
    features = [
        'value_eth', 'tx_fee', 'hour',
        'from_address_hashed_tx_count', 'to_address_hashed_tx_count',
        'value_z_score', 'gas', 'gas_price', 'time_since_last_sender_tx'
    ]
    X = df_proc[features].fillna(0).values
    print("🧠 Analyzing with Isolation Forest...")
    X_scaled = scaler.transform(X)

    df_proc['raw_score'] = model.decision_function(X_scaled)
    min_s, max_s = df_proc['raw_score'].min(), df_proc['raw_score'].max()
    # Protect against zero division
    denom = (max_s - min_s) if (max_s - min_s) != 0 else 1.0
    df_proc['risk_score'] = 100 * (1 - (df_proc['raw_score'] - min_s) / denom)
    df_proc['risk_score'] = df_proc['risk_score'].round(2)

    # 4. Save outputs
    df_proc.to_csv('all_scored_transactions.csv', index=False)
    print("✅ Saved all_scored_transactions.csv")

    top_100 = df_proc.sort_values(by='risk_score', ascending=False).head(100)
    report_cols = ['transaction_hash', 'value_eth', 'risk_score', 'block_timestamp', 'from_address_hashed']
    top_100[report_cols].to_csv('top_100_risky_transactions.csv', index=False)
    print("✅ Saved top_100_risky_transactions.csv")

    print("🕵️ Generating Watchlist...")
    df_high = df_proc[df_proc['risk_score'] >= RISK_THRESHOLD].copy()
    if not df_high.empty:
        # Normalize before aggregation
        df_high['from_address_hashed'] = df_high['from_address_hashed'].apply(normalize_hash)
        df_high['to_address_hashed'] = df_high['to_address_hashed'].apply(normalize_hash)

        sender_counts = df_high['from_address_hashed'].value_counts().reset_index()
        sender_counts.columns = ['wallet_hash', 'risky_tx_count']
        receiver_counts = df_high['to_address_hashed'].value_counts().reset_index()
        receiver_counts.columns = ['wallet_hash', 'risky_tx_count']

        # Normalize wallet_hash column too
        sender_counts['wallet_hash'] = sender_counts['wallet_hash'].apply(normalize_hash)
        receiver_counts['wallet_hash'] = receiver_counts['wallet_hash'].apply(normalize_hash)

        df_wallet_risk = pd.concat([sender_counts, receiver_counts]).groupby('wallet_hash').sum().reset_index()

        # Save watchlist with normalized wallet_hash
        df_watchlist = df_wallet_risk.sort_values('risky_tx_count', ascending=False).head(WATCHLIST_SIZE)
        df_watchlist['wallet_hash'] = df_watchlist['wallet_hash'].apply(normalize_hash)
        df_watchlist.to_csv('watchlist_accounts.csv', index=False)
        print("✅ Saved watchlist_accounts.csv")

        # Deep-dive: sender OR receiver, with direction (helps debugging)
        wl = [normalize_hash(w) for w in df_watchlist['wallet_hash'].tolist()]
        df_deep = df_proc[
            df_proc['from_address_hashed'].isin(wl) |
            df_proc['to_address_hashed'].isin(wl)
        ].copy()
        df_deep['direction'] = np.where(
            df_deep['from_address_hashed'].isin(wl), 'sent',
            np.where(df_deep['to_address_hashed'].isin(wl), 'received', 'other')
        )
        df_deep.to_csv('watchlist_deep_dive_report.csv', index=False)
        print("✅ Saved watchlist_deep_dive_report.csv")

        # NEW: per-wallet sender-only CSVs (safe filenames)
        os.makedirs(WALLET_HIST_DIR, exist_ok=True)
        kept_cols = ['block_timestamp', 'value_eth', 'risk_score']
        for w in wl:
            sent = df_proc[df_proc['from_address_hashed'] == w][kept_cols].copy()
            # convert timestamps to ISO strings for easy reading later
            sent['block_timestamp'] = pd.to_datetime(sent['block_timestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S%z')
            fname = safe_wallet_filename(w) + "_sent.csv"
            sent.to_csv(os.path.join(WALLET_HIST_DIR, fname), index=False)
        print(f"✅ Saved sender-only histories in ./{WALLET_HIST_DIR}/<prefix>_<sha1>_sent.csv")

    else:
        print("⚠️ No high-risk transactions found.")
        pd.DataFrame(columns=['wallet_hash', 'risky_tx_count']).to_csv('watchlist_accounts.csv', index=False)
        pd.DataFrame(columns=report_cols + ['direction']).to_csv('watchlist_deep_dive_report.csv', index=False)

    print("\n=== 🛡️ CHAINGUARD REPORT SUMMARY ===")
    print(tabulate(top_100[['transaction_hash', 'risk_score']].head(5), headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    generate_risk_report()
