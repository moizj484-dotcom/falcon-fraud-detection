
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 45)
print("       FALCON ADVANCED EDA")
print("=" * 45)

files = list(RAW_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError("Dataset not found in data/raw")

file_path = next(
    (f for f in files if "PS_20174392719" in f.name),
    files[0]
)

print("Loading:", file_path.name)

type_parts = []
fraud_parts = []

for chunk in pd.read_csv(file_path, chunksize=100000):

    type_summary = chunk.groupby("type").agg(
        total_transactions=("isFraud", "size"),
        total_fraud=("isFraud", "sum"),
        total_amount=("amount", "sum")
    ).reset_index()

    type_parts.append(type_summary)

    fraud_chunk = chunk[chunk["isFraud"] == 1]

    if not fraud_chunk.empty:
        fraud_summary = fraud_chunk.groupby("type").agg(
            fraud_transactions=("isFraud", "size"),
            total_fraud_amount=("amount", "sum")
        ).reset_index()

        fraud_parts.append(fraud_summary)

# Combine transaction summaries
type_result = pd.concat(type_parts, ignore_index=True)

type_result = type_result.groupby(
    "type", as_index=False
).sum(numeric_only=True)

# Combine fraud summaries safely
if fraud_parts:
    fraud_result = pd.concat(
        fraud_parts, ignore_index=True
    )

    fraud_result = fraud_result.groupby(
        "type", as_index=False
    ).sum(numeric_only=True)
else:
    fraud_result = pd.DataFrame(columns=[
        "type",
        "fraud_transactions",
        "total_fraud_amount"
    ])

# Merge summaries
final_result = type_result.merge(
    fraud_result,
    on="type",
    how="left"
)

final_result = final_result.fillna(0)

final_result["fraud_percentage"] = (
    final_result["total_fraud"]
    / final_result["total_transactions"]
) * 100

print("\nTRANSACTION TYPE ANALYSIS")
print(final_result.to_string(index=False))

# Save CSV
csv_path = OUTPUT_DIR / "transaction_type_summary.csv"
final_result.to_csv(csv_path, index=False)

# Generate graph
plt.figure(figsize=(10, 6))

plt.bar(
    final_result["type"],
    final_result["total_fraud"]
)

plt.title("FALCON - Fraud Transactions by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Fraud Transactions")
plt.tight_layout()

graph_path = OUTPUT_DIR / "fraud_by_transaction_type.png"
plt.savefig(graph_path)
plt.close()

print("\nADVANCED EDA COMPLETED")
print("Summary:", csv_path)
print("Graph:", graph_path)
