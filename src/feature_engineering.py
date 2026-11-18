"""
FALCON - Advanced Feature Engineering Pipeline
Constructs behavioral, temporal, and financial discrepancy features from PaySim data.
Maintains 100% of fraud transactions for optimal detection power while optimizing memory.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(RAW_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError("Raw PaySim dataset not found in data/raw")

DATA_PATH = next((f for f in files if "PS_20174392719" in f.name), files[0])

print("=" * 60)
print("       FALCON FEATURE ENGINEERING PIPELINE")
print("=" * 60)
print(f"Reading dataset: {DATA_PATH.name}")

cols_to_use = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud"
]

dtypes = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8"
}

df = pd.read_csv(DATA_PATH, usecols=cols_to_use, dtype=dtypes)

print(f"Raw records loaded: {len(df):,}")

# Clean and normalize type
df["type"] = df["type"].astype(str).str.upper()

# -------------------------------------------------------------
# Stratified Sample with 100% Fraud Preservation
# Total dataset is ~6.36M rows. We keep ALL 8,213 fraud instances
# and a representative sample of legitimate transactions for balanced,
# fast, and memory-safe model training (under 500k rows).
# -------------------------------------------------------------
fraud_df = df[df["isFraud"] == 1]
legit_df = df[df["isFraud"] == 0]

TARGET_LEGIT_SIZE = 300000
if len(legit_df) > TARGET_LEGIT_SIZE:
    legit_df = legit_df.sample(n=TARGET_LEGIT_SIZE, random_state=42)

df_sampled = pd.concat([fraud_df, legit_df], ignore_index=True)
df_sampled = df_sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)

del df, fraud_df, legit_df
print(f"Sampled working set: {len(df_sampled):,} (Preserved all {int(df_sampled['isFraud'].sum()):,} fraud records)")

# -------------------------------------------------------------
# 1. Temporal Features
# -------------------------------------------------------------
df_sampled["hour"] = (df_sampled["step"] % 24).astype("int8")
df_sampled["day"] = (df_sampled["step"] // 24).astype("int16")

# -------------------------------------------------------------
# 2. Transaction Amount Transformations
# -------------------------------------------------------------
df_sampled["log_amount"] = np.log1p(df_sampled["amount"].clip(lower=0)).astype("float32")

# -------------------------------------------------------------
# 3. Origin Account Discrepancy Features
# -------------------------------------------------------------
df_sampled["origin_balance_error"] = (
    df_sampled["oldbalanceOrg"] - df_sampled["newbalanceOrig"] - df_sampled["amount"]
).astype("float32")

df_sampled["origin_balance_change"] = (
    df_sampled["oldbalanceOrg"] - df_sampled["newbalanceOrig"]
).astype("float32")

# Origin account completely emptied to zero (classic fraud indicator)
df_sampled["origin_emptied"] = (
    (df_sampled["oldbalanceOrg"] > 0) & (df_sampled["newbalanceOrig"] == 0)
).astype("int8")

# -------------------------------------------------------------
# 4. Destination Account Discrepancy Features
# -------------------------------------------------------------
df_sampled["destination_balance_change"] = (
    df_sampled["newbalanceDest"] - df_sampled["oldbalanceDest"]
).astype("float32")

df_sampled["destination_balance_error"] = (
    df_sampled["newbalanceDest"] - df_sampled["oldbalanceDest"] - df_sampled["amount"]
).astype("float32")

# Destination had 0 balance initially and received sudden funds
df_sampled["dest_zero_init"] = (
    (df_sampled["oldbalanceDest"] == 0) & (df_sampled["newbalanceDest"] > 0)
).astype("int8")

# -------------------------------------------------------------
# 5. One-Hot Encoding Transaction Types
# -------------------------------------------------------------
allowed_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
for t in allowed_types:
    df_sampled[f"type_{t}"] = (df_sampled["type"] == t).astype("int8")

df_sampled = df_sampled.drop(columns=["type"])

# -------------------------------------------------------------
# 6. Cleaning & Validation
# -------------------------------------------------------------
df_sampled.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sampled.fillna(0, inplace=True)

OUTPUT_FILE = OUTPUT_DIR / "falcon_features.csv"
df_sampled.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("FALCON FEATURE ENGINEERING COMPLETED")
print(f"Output File:     {OUTPUT_FILE}")
print(f"Total Rows:      {len(df_sampled):,}")
print(f"Total Features:  {len(df_sampled.columns)}")
print(f"Fraud Rows:      {int(df_sampled['isFraud'].sum()):,}")
print(f"Feature Columns: {list(df_sampled.columns)}")
print("=" * 60)
