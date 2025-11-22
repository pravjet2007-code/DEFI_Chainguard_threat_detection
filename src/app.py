
import streamlit as st
import pandas as pd
import os
import hashlib

st.set_page_config(layout="wide", page_title="Chainguard Threat Intelligence")

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATHS = {
    "top_100": os.path.join(BASE_DIR, 'top_100_risky_transactions.csv'),
    "watchlist": os.path.join(BASE_DIR, 'watchlist_accounts.csv'),
    "deep_dive": os.path.join(BASE_DIR, 'watchlist_deep_dive_report.csv'),
    "wallet_histories_dir": os.path.join(BASE_DIR, 'wallet_histories')
}

RISK_THRESHOLD = 95  # for red styling

def normalize_hash(h):
    if h is None: return None
    return str(h).strip().lower()

def safe_wallet_filename(w_hash_normalized):
    prefix = w_hash_normalized[:16]
    digest = hashlib.sha1(w_hash_normalized.encode()).hexdigest()
    return f"{prefix}_{digest}_sent.csv"

# --- LOAD DATA ---
@st.cache_data
def load_csv(path):
    if not os.path.exists(path): return None, "File not found"
    try:
        df = pd.read_csv(path)
        return df, "Success"
    except Exception as e:
        return None, str(e)

df_top_100, _ = load_csv(FILE_PATHS["top_100"])
df_watchlist, _ = load_csv(FILE_PATHS["watchlist"])
df_deep_dive, _ = load_csv(FILE_PATHS["deep_dive"])

if df_top_100 is None or df_watchlist is None or df_deep_dive is None:
    st.error("🚨 Data Missing! Please run: `python -m src.predict_and_report`")
    st.stop()

# Normalize wallet_hash column defensively
if "wallet_hash" in df_watchlist.columns:
    df_watchlist["wallet_hash"] = df_watchlist["wallet_hash"].apply(normalize_hash)

# Parse timestamp/risk types safely
for df in [df_top_100, df_deep_dive]:
    if "block_timestamp" in df.columns:
        df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], errors='coerce', utc=True)
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors='coerce')

# --- SESSION STATE: per-row open/close states ---
if "opened_rows" not in st.session_state:
    st.session_state.opened_rows = {}  # dict: wallet_hash -> bool

st.title("🛡️ Chainguard Threat Intelligence Dashboard")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚠️ Watchlist Accounts")

    if df_watchlist.empty:
        st.success("No high-risk accounts detected.")
    else:
        st.info(f"Monitoring {len(df_watchlist)} Suspicious Wallets")

        # Show more digits in preview label
        df_watchlist["preview"] = df_watchlist["wallet_hash"].apply(lambda h: f"{h[:16]}...")

        with st.expander("Inspect wallets (click ▶ to open transactions)", expanded=True):
            for _, row in df_watchlist.iterrows():
                wallet_hash = normalize_hash(row["wallet_hash"])
                alerts = int(row.get("risky_tx_count", 0))

                # Default toggle state
                if wallet_hash not in st.session_state.opened_rows:
                    st.session_state.opened_rows[wallet_hash] = False

                # Unique button key based on sha1 of normalized hash
                btn_key = "btn_open_" + hashlib.sha1(wallet_hash.encode()).hexdigest()[:12]

                # Row layout
                line = st.columns([4, 1])
                with line[0]:
                    st.write(f"`{row['preview']}`  ({alerts} alerts)")
                with line[1]:
                    if st.button("▶", key=btn_key):
                        st.session_state.opened_rows[wallet_hash] = not st.session_state.opened_rows[wallet_hash]

                # Container under the row for its table
                row_container = st.container()
                if st.session_state.opened_rows[wallet_hash]:
                    # Prefer per-wallet sender-only file
                    per_wallet_file = safe_wallet_filename(wallet_hash)
                    per_wallet_path = os.path.join(FILE_PATHS["wallet_histories_dir"], per_wallet_file)

                    if os.path.exists(per_wallet_path):
                        wallet_tx = pd.read_csv(per_wallet_path)
                        # Coerce types
                        if "block_timestamp" in wallet_tx.columns:
                            wallet_tx["block_timestamp"] = pd.to_datetime(wallet_tx["block_timestamp"], errors='coerce', utc=True)
                        if "risk_score" in wallet_tx.columns:
                            wallet_tx["risk_score"] = pd.to_numeric(wallet_tx["risk_score"], errors='coerce')
                        if "value_eth" in wallet_tx.columns:
                            wallet_tx["value_eth"] = pd.to_numeric(wallet_tx["value_eth"], errors='coerce')
                    else:
                        # Fallback: filter deep_dive (sender-only)
                        wallet_tx = df_deep_dive[df_deep_dive["from_address_hashed"].apply(normalize_hash) == wallet_hash].copy()

                    # Sort ascending by time
                    if "block_timestamp" in wallet_tx.columns:
                        wallet_tx = wallet_tx.sort_values("block_timestamp", ascending=True)

                    # If still empty -> friendly message
                    if wallet_tx.empty:
                        row_container.markdown(
                            f"**Wallet:** `{wallet_hash}` &nbsp;&nbsp; "
                            f"<span style='color:#16a34a; font-weight:bold;'>Avg Risk Score: 0.00</span>",
                            unsafe_allow_html=True
                        )
                        row_container.info("No sender transactions found for this wallet.")
                        continue

                    # Avg risk score (sender-only)
                    avg_risk = float(wallet_tx["risk_score"].astype(float).mean())

                    # Header line: full hash + Avg Risk in green
                    row_container.markdown(
                        f"**Wallet:** `{wallet_hash}`  &nbsp;&nbsp; "
                        f"<span style='color:#16a34a; font-weight:bold;'>Avg Risk Score: {avg_risk:.2f}</span>",
                        unsafe_allow_html=True
                    )

                    # Display required columns
                    show_cols = ["block_timestamp", "value_eth", "risk_score"]
                    show_cols = [c for c in show_cols if c in wallet_tx.columns]
                    view = wallet_tx[show_cols].copy()

                    # Style risky rows red
                    def style_risky(r):
                        try:
                            is_risky = float(r["risk_score"]) >= RISK_THRESHOLD
                        except Exception:
                            is_risky = False
                        return ["color: red; background-color: #ffecec;" if is_risky else ""] * len(r)

                    row_container.dataframe(
                        view.style.apply(style_risky, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )

with col_right:
    st.subheader("🔥 Top 100 Global Risk Alerts")
    if not df_top_100.empty:
        # Coerce
        df_top_100["risk_score"] = pd.to_numeric(df_top_100.get("risk_score"), errors='coerce')
        df_top_100["block_timestamp"] = pd.to_datetime(df_top_100.get("block_timestamp"), errors='coerce', utc=True)

        desired_order = ["transaction_hash", "value_eth", "risk_score", "block_timestamp", "from_address_hashed"]
        cols_present = [c for c in desired_order if c in df_top_100.columns]
        df_top_100_view = df_top_100[cols_present].copy()

        st.dataframe(
            df_top_100_view,
            column_config={
                "transaction_hash": "Transaction ID",
                "value_eth": "Value (ETH)",
                "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=100),
                "block_timestamp": "Timestamp",
                "from_address_hashed": "From (hashed)",
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transactions found.")
