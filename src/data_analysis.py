"""
FALCON - Exploratory Data Analysis & Statistical Profiling
Analyzes the PaySim dataset for fraud distribution, transaction patterns,
and financial discrepancies, saving statistical tables and visual charts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(RAW_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError("PaySim CSV dataset not found in data/raw")

DATA_PATH = next((f for f in files if "PS_20174392719" in f.name), files[0])

print("=" * 60)
print("       FALCON EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print(f"Loading Dataset: {DATA_PATH.name}")

columns = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    dtype={
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8"
    }
)

if df.empty:
    raise ValueError("Dataset is empty.")

total_records = len(df)
fraud_mask = df["isFraud"] == 1
total_fraud = int(fraud_mask.sum())
total_legit = total_records - total_fraud
fraud_rate_overall = (total_fraud / total_records) * 100

print(f"\nTotal Transactions:      {total_records:,}")
print(f"Legitimate Transactions: {total_legit:,} ({100 - fraud_rate_overall:.3f}%)")
print(f"Fraudulent Transactions: {total_fraud:,} ({fraud_rate_overall:.3f}%)")
print(f"Missing Values:          {df.isnull().sum().sum()}")

# -------------------------------------------------------------
# 1. Transaction Types Distribution
# -------------------------------------------------------------
type_counts = df["type"].value_counts()
type_counts.to_csv(OUTPUT_DIR / "transaction_types.csv")

plt.figure(figsize=(9, 5))
bars = plt.bar(type_counts.index, type_counts.values, color="#1e3a8a", edgecolor="#3b82f6")
plt.title("Transaction Volume by Type", fontsize=13, fontweight="bold")
plt.xlabel("Transaction Type", fontsize=11)
plt.ylabel("Number of Transactions", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:,.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_types.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 2. Fraud vs Legitimate Counts
# -------------------------------------------------------------
fraud_counts = pd.Series([total_legit, total_fraud], index=[0, 1])
fraud_counts.to_csv(OUTPUT_DIR / "fraud_counts.csv")

plt.figure(figsize=(7, 5))
bars = plt.bar(["Legitimate", "Fraud"], [total_legit, total_fraud], color=["#10b981", "#ef4444"])
plt.yscale("log")
plt.title("Fraud vs Legitimate Transactions (Log Scale)", fontsize=13, fontweight="bold")
plt.ylabel("Count (Log Scale)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
for bar, val in zip(bars, [total_legit, total_fraud]):
    plt.text(bar.get_x() + bar.get_width() / 2, val * 1.2, f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_distribution.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 3. Fraud Breakdown by Transaction Type
# -------------------------------------------------------------
type_summary = df.groupby("type", observed=True).agg(
    total_transactions=("isFraud", "size"),
    fraud_transactions=("isFraud", "sum"),
    total_amount=("amount", "sum")
).reset_index()

fraud_df = df[df["isFraud"] == 1]
fraud_amount_summary = fraud_df.groupby("type", observed=True).agg(
    total_fraud_amount=("amount", "sum")
).reset_index()

type_summary = type_summary.merge(fraud_amount_summary, on="type", how="left").fillna(0)
type_summary["fraud_percentage"] = (type_summary["fraud_transactions"] / type_summary["total_transactions"]) * 100
type_summary.to_csv(OUTPUT_DIR / "transaction_type_summary.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(type_summary["type"], type_summary["fraud_percentage"], color="#dc2626", edgecolor="#991b1b")
plt.title("Fraud Rate (%) by Transaction Type", fontsize=13, fontweight="bold")
plt.xlabel("Transaction Type", fontsize=11)
plt.ylabel("Fraud Percentage (%)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
for bar, val in zip(bars, type_summary["fraud_percentage"]):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_rate_by_type.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
bars = plt.bar(type_summary["type"], type_summary["fraud_transactions"], color="#b91c1c")
plt.title("Fraud Transactions Count by Type", fontsize=13, fontweight="bold")
plt.xlabel("Transaction Type", fontsize=11)
plt.ylabel("Fraud Count", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
for bar, val in zip(bars, type_summary["fraud_transactions"]):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_by_transaction_type.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 4. Transaction Amount Log Distribution
# -------------------------------------------------------------
sample_legit = df.loc[~fraud_mask, "amount"].sample(n=min(total_legit, 100000), random_state=42)
sample_fraud = df.loc[fraud_mask, "amount"]

plt.figure(figsize=(10, 5))
plt.hist(np.log1p(sample_legit), bins=50, alpha=0.55, label="Legitimate", color="#3b82f6", density=True)
plt.hist(np.log1p(sample_fraud), bins=50, alpha=0.65, label="Fraud", color="#ef4444", density=True)
plt.title("Transaction Amount Distribution (Log Scale)", fontsize=13, fontweight="bold")
plt.xlabel("Log(1 + Amount)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_amount.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 5. Origin Balance Error Analysis
# -------------------------------------------------------------
df["origin_balance_error"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
)

sample_legit_err = df.loc[~fraud_mask, "origin_balance_error"].sample(n=min(total_legit, 10000), random_state=42)
sample_fraud_err = df.loc[fraud_mask, "origin_balance_error"].sample(n=min(total_fraud, 10000), random_state=42)

plt.figure(figsize=(8, 5))
# Compatible with Matplotlib 3.9+ / 3.11+
plt.boxplot(
    [sample_legit_err, sample_fraud_err],
    tick_labels=["Legitimate", "Fraud"],
    showfliers=False,
    patch_artist=True
)
plt.title("Origin Balance Discrepancy Error", fontsize=13, fontweight="bold")
plt.ylabel("Discrepancy (Old - New - Amount)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "balance_error.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 6. Hourly Pattern of Fraud
# -------------------------------------------------------------
df["hour"] = df["step"] % 24
hourly_summary = df.groupby("hour", observed=True)["isFraud"].agg(["size", "sum", "mean"]).reset_index()
hourly_summary["fraud_rate_pct"] = hourly_summary["mean"] * 100

plt.figure(figsize=(10, 5))
plt.plot(hourly_summary["hour"], hourly_summary["fraud_rate_pct"], marker="o", color="#dc2626", linewidth=2.5)
plt.title("Fraud Occurrence Rate by Hour of Day", fontsize=13, fontweight="bold")
plt.xlabel("Hour (0 - 23)", fontsize=11)
plt.ylabel("Fraud Rate (%)", fontsize=11)
plt.xticks(range(0, 24))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hourly_fraud_pattern.png", dpi=150)
plt.close()

# -------------------------------------------------------------
# 7. Summary Report
# -------------------------------------------------------------
summary = pd.DataFrame({
    "Category": ["Legitimate", "Fraud", "Total"],
    "Transactions": [total_legit, total_fraud, total_records],
    "Percentage": [f"{(total_legit / total_records) * 100:.3f}%", f"{fraud_rate_overall:.3f}%", "100.000%"]
})
summary.to_csv(OUTPUT_DIR / "eda_summary.csv", index=False)

print("\n" + "=" * 60)
print("FALCON EDA COMPLETED SUCCESSFULLY")
print(f"Generated 6 Visualizations and 4 Summary Tables in: {OUTPUT_DIR}")
print("=" * 60)
