import pandas as pd
import os

# --- Configuration ---
RISK_THRESHOLD = 95  # Minimum risk score to qualify as "high-risk"
WATCHLIST_SIZE = 50  # Number of accounts to put on the final watchlist

def generate_watchlists():
    
    print("Starting Watchlist generation...")
    
    # 1. Load ALL scored data
    data_path = 'all_scored_transactions.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Run predict_and_report.py first.")
        return
        
    df_all = pd.read_csv(data_path)

    # 2. Identify High-Risk Transactions
    df_high_risk = df_all[df_all['risk_score'] >= RISK_THRESHOLD].copy()
    
    # 3. Aggregate Risky Transactions by Wallet
    # Count how many high-risk transactions each wallet (sender or receiver) has.
    
    # Combine sender and receiver lists for aggregation
    sender_counts = df_high_risk.groupby('from_address_hashed')['risk_score'].count().reset_index(name='risky_tx_count')
    receiver_counts = df_high_risk.groupby('to_address_hashed')['risk_score'].count().reset_index(name='risky_tx_count')
    
    # Rename columns to a generic 'wallet_hash'
    sender_counts = sender_counts.rename(columns={'from_address_hashed': 'wallet_hash'})
    receiver_counts = receiver_counts.rename(columns={'to_address_hashed': 'wallet_hash'})
    
    # Combine and sum the counts (a wallet can be both sender and receiver)
    df_wallet_risk = pd.concat([sender_counts, receiver_counts])
    df_wallet_risk = df_wallet_risk.groupby('wallet_hash')['risky_tx_count'].sum().reset_index()
    
    # 4. Create the Watchlist (Top N Accounts)
    df_watchlist = df_wallet_risk.sort_values('risky_tx_count', ascending=False).head(WATCHLIST_SIZE)
    df_watchlist['risk_level'] = pd.cut(df_watchlist['risky_tx_count'], bins=3, labels=['Low', 'Medium', 'High'])
    
    df_watchlist.to_csv('watchlist_accounts.csv', index=False)
    print(f"✅ Watchlist created with {len(df_watchlist)} accounts.")
    
    # 5. Deep Dive Report (Fetch ALL transactions for these wallets)
    # Get a list of the top wallets
    watchlist_hashes = df_watchlist['wallet_hash'].tolist()
    
    # Select all transactions where the sender OR receiver is on the watchlist
    df_deep_dive = df_all[
        df_all['from_address_hashed'].isin(watchlist_hashes) | 
        df_all['to_address_hashed'].isin(watchlist_hashes)
    ]
    
    # Merge the watchlist data (risky_tx_count) onto the deep dive transactions
    df_deep_dive = pd.merge(df_deep_dive, df_watchlist[['wallet_hash', 'risky_tx_count']], 
                             left_on='from_address_hashed', right_on='wallet_hash', 
                             how='left').drop(columns=['wallet_hash']).fillna({'risky_tx_count': 0})
    
    df_deep_dive.to_csv('watchlist_deep_dive_report.csv', index=False)
    print("✅ Deep Dive Report (ALL transactions for watchlist accounts) saved.")

if __name__ == '__main__':
    generate_watchlists()