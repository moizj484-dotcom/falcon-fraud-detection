"""
FALCON - Real-Time Fraud Prediction Engine
Loads the trained ML model, transforms incoming transaction data,
computes risk probabilities, and provides fraud explainability flags.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "falcon_model.pkl"
FEATURE_PATH = ROOT / "models" / "feature_names.json"

ALLOWED_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_model():
    """Loads the serialized model and registered feature names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature names file not found at: {FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    return model, feature_names


def analyze_transaction_flags(step, transaction_type, amount, old_org, new_org, old_dest, new_dest):
    """Generates explainability flags for why a transaction is flagged suspicious."""
    flags = []
    
    # 1. Type check
    if transaction_type in ["TRANSFER", "CASH_OUT"]:
        flags.append(f"High-risk transaction channel: {transaction_type}")
    
    # 2. Account draining check
    if old_org > 0 and new_org == 0:
        flags.append("Origin account completely drained to 0 balance")
        
    # 3. Destination zero initial balance
    if old_dest == 0 and new_dest > 0:
        flags.append("Destination account had zero initial balance before transfer")
        
    # 4. Large amount check
    if amount >= 200000:
        flags.append(f"Unusually large transaction amount (${amount:,.2f})")
        
    # 5. Balance mismatch error
    discrepancy = old_org - new_org - amount
    if abs(discrepancy) > 1.0:
        flags.append(f"Significant origin balance discrepancy (${abs(discrepancy):,.2f})")
        
    # 6. Night hour transaction
    hour = int(step) % 24
    if hour in range(0, 6):
        flags.append(f"Unusual late-night transaction hour ({hour:02d}:00)")

    return flags


def predict_transaction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    """
    Evaluates a transaction for financial fraud probability.
    Returns:
        dict: prediction status, probability %, risk level, and explainability flags.
    """
    model, feature_names = load_model()

    # Input validation
    values = [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError("All balance and amount inputs must be valid numbers.")

    if not all(np.isfinite(v) for v in values):
        raise ValueError("Transaction values must be finite numbers.")

    if any(v < 0 for v in values):
        raise ValueError("Transaction amounts and balances cannot be negative.")

    step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest = values
    step = int(step)
    transaction_type = str(transaction_type).strip().upper()

    if transaction_type not in ALLOWED_TYPES:
        raise ValueError(f"Invalid transaction type '{transaction_type}'. Allowed: {ALLOWED_TYPES}")

    # Feature transformation
    hour = int(step) % 24
    day = int(step) // 24
    log_amount = float(np.log1p(max(0.0, amount)))
    origin_balance_error = float(oldbalanceOrg - newbalanceOrig - amount)
    origin_balance_change = float(oldbalanceOrg - newbalanceOrig)
    origin_emptied = int(oldbalanceOrg > 0 and newbalanceOrig == 0)
    destination_balance_change = float(newbalanceDest - oldbalanceDest)
    destination_balance_error = float(newbalanceDest - oldbalanceDest - amount)
    dest_zero_init = int(oldbalanceDest == 0 and newbalanceDest > 0)

    data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "hour": hour,
        "day": day,
        "log_amount": log_amount,
        "origin_balance_error": origin_balance_error,
        "origin_balance_change": origin_balance_change,
        "origin_emptied": origin_emptied,
        "destination_balance_change": destination_balance_change,
        "destination_balance_error": destination_balance_error,
        "dest_zero_init": dest_zero_init
    }

    for t in ALLOWED_TYPES:
        data[f"type_{t}"] = int(transaction_type == t)

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    fraud_prob = float(probabilities[1])

    # Determine risk category
    if fraud_prob >= 0.70:
        risk_level = "CRITICAL RISK"
    elif fraud_prob >= 0.40:
        risk_level = "HIGH RISK"
    elif fraud_prob >= 0.15:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    flags = analyze_transaction_flags(
        step, transaction_type, amount,
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest
    )

    return {
        "prediction": "FRAUD" if prediction == 1 or fraud_prob >= 0.50 else "LEGITIMATE",
        "fraud_probability": round(fraud_prob * 100, 2),
        "confidence": round(max(probabilities) * 100, 2),
        "risk_level": risk_level,
        "flags": flags
    }


if __name__ == "__main__":
    print("=" * 60)
    print("       FALCON PREDICTION ENGINE TEST")
    print("=" * 60)

    test_legit = predict_transaction(
        step=50,
        transaction_type="PAYMENT",
        amount=2500.0,
        oldbalanceOrg=10000.0,
        newbalanceOrig=7500.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0
    )

    print("\nTEST 1 - Legitimate Payment:")
    print(f"  Prediction:  {test_legit['prediction']}")
    print(f"  Probability: {test_legit['fraud_probability']}%")
    print(f"  Risk Level:  {test_legit['risk_level']}")

    test_fraud = predict_transaction(
        step=100,
        transaction_type="TRANSFER",
        amount=500000.0,
        oldbalanceOrg=500000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0
    )

    print("\nTEST 2 - Account Draining Transfer (Fraud):")
    print(f"  Prediction:  {test_fraud['prediction']}")
    print(f"  Probability: {test_fraud['fraud_probability']}%")
    print(f"  Risk Level:  {test_fraud['risk_level']}")
    print(f"  Flags Raised: {test_fraud['flags']}")
    print("=" * 60)
