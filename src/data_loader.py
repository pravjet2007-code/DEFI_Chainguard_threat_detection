import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes data is in the 'data' folder at the root level
csv_path = os.path.join(current_dir, '..', 'data', 'ethereum_sample_data.csv')

def load_data():
    """Loads the Ethereum transaction data."""
    if not os.path.exists(csv_path):
        print(f" ERROR: File not found at {csv_path}. Did you place the CSV in the 'data' folder?")
        return None
    
    df = pd.read_csv(csv_path)
    print(f" SUCCESS: Data Loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

if __name__ == "__main__":
    load_data()