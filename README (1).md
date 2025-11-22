
# 🛡️ Chainguard Threat Intelligence (MVP)

Chainguard is a lightweight, end‑to‑end **blockchain risk analysis MVP** designed for hackathon demos and rapid investigator workflows. It ingests transaction CSVs, **hashes addresses** for privacy, **extracts features**, generates **risk scores** (via model or heuristic fallback), and presents results in an attractive **Streamlit dashboard** with a **Wallet Explorer** and **Top Alerts**. A **Jupyter notebook** mirrors the pipeline for transparent, step‑by‑step demonstrations (including rich visualizations).

---

## ✨ Highlights

- **Privacy‑preserving hashing** of wallet addresses (salted SHA‑256).
- **Feature engineering**: value in ETH, tx fee, hour of day, per‑address tx counts, sender z‑score, time since last sender tx.
- **Risk scoring**:
  - **Model‑based** (Isolation Forest) if `chainguard_if_model.pkl` and `chainguard_scaler.pkl` are available.
  - **Heuristic fallback** (|z| + normalized fee) when models aren’t present—ideal for hackathon demos.
- **Artifacts** saved for interoperability with the UI:
  - `all_scored_transactions.csv`
  - `top_100_risky_transactions.csv`
  - `watchlist_accounts.csv`
  - `watchlist_deep_dive_report.csv`
- **Streamlit UI**:
  - **Wallet Explorer** tab: search, Sent/Received/Both toggle, inline ▶ expand, KPI chips, CSV export.
  - **Top 100 Alerts** tab: ordered columns + risk progress bar.
  - **Overview** tab: KPIs + risk distribution chart.
  - **Upload & Analyze** tab (optional): judges can upload CSVs and see insights instantly.

---

## 🗂️ Folder Structure (recommended)

```
Chainguard/
├─ app.py                          # Streamlit dashboard (UI-only)
├─ src/
│  ├─ predict_and_report.py        # Pipeline: features → risk → artifacts
│  ├─ watchlist_generator.py       # (optional) standalone watchlist tool
│  └─ data_loader.py               # Your data loader (imported in pipeline)
├─ models/
│  ├─ chainguard_if_model.pkl      # Isolation Forest (optional)
│  └─ chainguard_scaler.pkl        # Scaler used with the model (optional)
├─ notebooks/
│  ├─ Chainguard_MVP_Demo_EthereumSample.ipynb   # tailored demo notebook
│  └─ Chainguard_MVP_Demo_Visualized.ipynb       # extra visualization notebook
├─ all_scored_transactions.csv     # generated
├─ top_100_risky_transactions.csv  # generated
├─ watchlist_accounts.csv          # generated
├─ watchlist_deep_dive_report.csv  # generated
└─ README.md
```

> You can keep notebooks and UI in the repo root for convenience during demos; just ensure `BASE_DIR` in `app.py` points one level above `src/`.

---

## ⚙️ Environment & Requirements

- **Python** ≥ 3.9
- **Packages**: `pandas`, `numpy`, `joblib`, `streamlit`, `matplotlib` (+ `seaborn` optional), `scikit-learn` (if you plan to train/use the model).

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
pandas
numpy
joblib
streamlit
matplotlib
seaborn
scikit-learn
```

---

## 🔗 Input Data Schema

Your pipeline expects a CSV with (best‑effort mapping handles common aliases):

| Column               | Aliases                         | Notes                                       |
|----------------------|----------------------------------|---------------------------------------------|
| `block_timestamp`    | `timestamp`, `time`              | ISO or parseable datetime                   |
| `transaction_hash`   | `tx_hash`, `hash`                | Unique tx id                                |
| `from_address`       | `from`                           | Sender address (plain)                      |
| `to_address`         | `to`                             | Receiver address (plain)                    |
| `value`              | `value_wei`, `amount`            | Value in **wei** (numeric or scientific notation) |
| `gas`                |                                  | Numeric                                     |
| `gas_price`          |                                  | Numeric                                     |

> The notebook and pipeline will **coerce types** with `errors='coerce'` and map aliases automatically.

---

## 🔬 How the Risk Scoring Works

### Feature Engineering (in `predict_and_report.py` / notebooks)
- `value_eth = value / 10**18`
- `tx_fee  = gas * gas_price`
- `hour    = block_timestamp.dt.hour`
- `from_address_hashed`, `to_address_hashed` via **salted SHA‑256** (privacy‑preserving).
- Per‑wallet counts:
  - `from_address_hashed_tx_count`
  - `to_address_hashed_tx_count`
- Sender history stats:
  - `sender_mean`, `sender_std` → `value_z_score` (with NaN/Inf fixed via `np.nan_to_num`)
- Temporal:
  - `time_since_last_sender_tx` via per‑sender diffs in seconds.

### Model or Heuristic
- **Model**: Isolation Forest + scaler (`decision_function → min/max normalization → 0..100` risk; inverted so higher = riskier).
- **Heuristic** (fallback):  
  `risk = 0.6 * |value_z_score| + 0.4 * normalized(tx_fee)` → min/max normalized to `0..100`.

---

## 🧪 Producing Artifacts (CLI)

Run the pipeline:

```bash
python -m src.predict_and_report
```

Outputs created in the project root:

- `all_scored_transactions.csv` — full scored dataset.
- `top_100_risky_transactions.csv` — ordered columns:
  ```
  transaction_hash, value_eth, risk_score, block_timestamp, from_address_hashed
  ```
- `watchlist_accounts.csv` — top wallets by count of **high‑risk** events (threshold = 95 by default).
- `watchlist_deep_dive_report.csv` — all txns involving watchlist wallets; includes `direction` (sent/received/other).

> If the dataset yields no high‑risk events, **empty files** are still created to prevent UI errors.

---

## 🖥️ Streamlit Dashboard (UI‑only)

Start the app:

```bash
streamlit run app.py
```

### Tabs & Features

- **Overview**
  - KPIs: watchlist count, deep‑dive tx count, average risk, high‑risk event count.
  - Risk distribution (Top‑100) bar chart.

- **Wallet Explorer**
  - **Search** by hash prefix (case‑insensitive).
  - Toggle **Sent / Received / Both**.
  - Inline **▶** button per wallet row:
    - Expands a table (Timestamp, Value (ETH), Risk).
    - KPI chips: **Avg Risk** (green), **Max Risk**, **Last Seen**, **Total Tx**.
    - **Download CSV** for the visible subset.

- **Top 100 Alerts**
  - Ordered table: **Transaction ID, Value (ETH), Risk Score, Timestamp, From (hashed)**.
  - Risk score displayed as a **progress bar**.

- **Upload & Analyze** (optional tab)
  - Upload a CSV; auto‑map columns; score with model if available, else heuristic.
  - Shows Top Alerts and a scoped Wallet Explorer for the uploaded dataset.
  - No persistence needed; adds **API‑like feel** without backend risk.

> The UI uses defensive parsing and string normalization to avoid key collisions and empty views for specific wallets.

---

## 📓 Demonstration Notebook

Use the notebook tailored to your dataset:

- **`notebooks/Chainguard_MVP_Demo_EthereumSample.ipynb`**  
  Walks through: load → hashing → features → risk → artifacts → **rich visualizations** (hist/KDE, box/violin by hour, scatter, time series, correlation heatmap, top wallets).

### Running the Notebook
1. Open the notebook in VS Code/Jupyter.
2. If your CSV is not in the repo root, set the absolute path in the **config cell**:
   ```python
   DATA_PATH = r"C:\\Users\\YourName\\Downloads\\ethereum_sample_data.csv"
   ```
3. If the model artifacts exist in `./models/`, the notebook uses them; otherwise it falls back gracefully.

---

## 🔧 Configuration Snippets

**Using `pathlib` (recommended):**
```python
from pathlib import Path

BASE_DIR   = Path('.').resolve()
MODEL_DIR  = BASE_DIR / 'models'

MODEL_PATH  = MODEL_DIR / 'chainguard_if_model.pkl'
SCALER_PATH = MODEL_DIR / 'chainguard_scaler.pkl'

OUT_ALL_SCORED = BASE_DIR / 'all_scored_transactions.csv'
OUT_TOP100     = BASE_DIR / 'top_100_risky_transactions.csv'
OUT_WATCHLIST  = BASE_DIR / 'watchlist_accounts.csv'
OUT_DEEPDIVE   = BASE_DIR / 'watchlist_deep_dive_report.csv'

RISK_THRESHOLD = 95
WATCHLIST_SIZE = 50
```

---

## 🧰 Troubleshooting

- **FileNotFoundError for CSV in notebook**  
  Fix `DATA_PATH` to the absolute file path or move the file next to the notebook. A robust load cell with glob search or an upload widget can be used if preferred.

- **Models not found**  
  Ensure `models/chainguard_if_model.pkl` and `models/chainguard_scaler.pkl` exist. If missing, the pipeline and notebook **will still work** via the heuristic.

- **Streamlit shows empty wallet table**  
  Some wallets are watchlisted due to **received** events; sender‑only subset may be empty. Switch the Explorer toggle to **Received/Both** to view inbound activity.

- **Button/row expansion works for only one wallet**  
  Fixed by using **unique keys** (sha1 of wallet hash) and per‑wallet session state maps in `app.py`.

---

## 🔒 Security & Privacy

- Addresses are **salted** and hashed (`SHA‑256`) before analysis, preventing direct exposure in dashboards and saved artifacts.
- Avoid storing plaintext addresses in permanent artifacts; rely on hashed versions for reporting.

---

## 🗺️ Roadmap (suggested)

- Severity bands & badges (High/Severe/Critical).
- Per‑wallet sparklines of risk over time.
- Lightweight case export: bundle CSVs + JSON summary for investigators.
- Model retraining notebook + hyperparameter tuning.
- Optional per‑wallet `sent` / `received` CSV caching for faster UI loading on large datasets.

---

## 📝 License

This MVP is intended for hackathon demonstration and research use. Choose a license (e.g., MIT) that suits your team’s goals.

---

## 🙌 Acknowledgements

Built by **Sharma Parvinkumar Madanlal** as part of a hackathon MVP. Copilot assisted with pipeline hardening, UI polish, and notebook generation.

---

## 🚀 Quickstart Summary

```bash
# 1) Prepare environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Generate artifacts
python -m src.predict_and_report

# 3) Run the dashboard
streamlit run app.py

# 4) (Optional) Demo with the notebook
# Open notebooks/Chainguard_MVP_Demo_EthereumSample.ipynb and run all cells
```
