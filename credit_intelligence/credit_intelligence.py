"""
Credit Intelligence - simple base project
-----------------------------------------
Predicts the probability that an applicant will default, converts it into a
300-850 credit score, assigns a risk band, and lists the factors driving it.

Usage:
    pip install pandas numpy scikit-learn joblib
    python credit_intelligence.py train                 # train + evaluate (synthetic data)
    python credit_intelligence.py train --csv my.csv    # train on your own data
    python credit_intelligence.py predict               # interactive applicant scoring

Your CSV must contain the FEATURES columns below plus a 0/1 column named "default".
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "credit_model.joblib"

FEATURES = {
    "age": "Age (years)",
    "annual_income": "Annual income",
    "loan_amount": "Requested loan amount",
    "existing_debt": "Existing total debt",
    "credit_history_years": "Length of credit history (years)",
    "late_payments": "Number of late payments (last 2 yrs)",
    "credit_utilization": "Credit utilization (0.0 - 1.0)",
    "employment_years": "Years in current employment",
}


# ---------------------------------------------------------------- data ----
def generate_synthetic_data(n=5000, seed=42):
    """Create realistic-looking applicant data with a hidden default rule."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "annual_income": rng.lognormal(10.8, 0.5, n).round(0),
        "loan_amount": rng.lognormal(9.5, 0.7, n).round(0),
        "existing_debt": rng.lognormal(9.0, 1.0, n).round(0),
        "credit_history_years": rng.integers(0, 30, n),
        "late_payments": rng.poisson(1.0, n),
        "credit_utilization": rng.beta(2, 3, n).round(2),
        "employment_years": rng.integers(0, 35, n),
    })
    risk = (
        -2.0
        + 1.8 * (df.loan_amount / df.annual_income)
        + 1.2 * (df.existing_debt / df.annual_income)
        + 0.45 * df.late_payments
        + 1.5 * df.credit_utilization
        - 0.05 * df.credit_history_years
        - 0.03 * df.employment_years
        + rng.normal(0, 0.5, n)
    )
    df["default"] = (1 / (1 + np.exp(-risk)) > 0.5).astype(int)
    return df


def add_ratios(df):
    """Derived features that matter a lot in credit risk."""
    out = df.copy()
    out["loan_to_income"] = out["loan_amount"] / out["annual_income"].clip(lower=1)
    out["debt_to_income"] = out["existing_debt"] / out["annual_income"].clip(lower=1)
    return out


# ------------------------------------------------------------- scoring ----
def prob_to_score(p_default):
    """Map default probability to a 300-850 score (higher = safer)."""
    return int(round(850 - 550 * np.clip(p_default, 0, 1)))


def risk_band(score):
    if score >= 750:
        return "Excellent (Very Low Risk)"
    if score >= 680:
        return "Good (Low Risk)"
    if score >= 600:
        return "Fair (Medium Risk)"
    if score >= 500:
        return "Poor (High Risk)"
    return "Very Poor (Very High Risk)"


def decision(score):
    if score >= 680:
        return "APPROVE"
    if score >= 600:
        return "MANUAL REVIEW"
    return "DECLINE"


# --------------------------------------------------------------- train ----
def train(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        df = generate_synthetic_data()
        print(f"Generated {len(df)} synthetic applicants")

    X = add_ratios(df[list(FEATURES)])
    y = df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"\nROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, model.predict(X_te), digits=3))

    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top drivers of default risk:")
    print(importances.sort_values(ascending=False).head(5).round(3).to_string())

    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


# ------------------------------------------------------------- predict ----
def score_applicant(applicant: dict):
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["columns"]

    X = add_ratios(pd.DataFrame([applicant]))[cols]
    p = float(model.predict_proba(X)[0, 1])
    score = prob_to_score(p)

    reasons = explain(applicant)
    return {
        "default_probability": round(p, 3),
        "credit_score": score,
        "risk_band": risk_band(score),
        "decision": decision(score),
        "key_factors": reasons,
    }


def explain(a):
    """Simple rule-based reason codes (readable by non-technical users)."""
    reasons = []
    if a["late_payments"] >= 2:
        reasons.append("Multiple recent late payments")
    if a["credit_utilization"] > 0.6:
        reasons.append("High credit utilization")
    if a["existing_debt"] / max(a["annual_income"], 1) > 0.5:
        reasons.append("High debt-to-income ratio")
    if a["loan_amount"] / max(a["annual_income"], 1) > 1.0:
        reasons.append("Loan amount is large relative to income")
    if a["credit_history_years"] < 3:
        reasons.append("Short credit history")
    if a["employment_years"] < 1:
        reasons.append("Limited employment stability")
    return reasons or ["No major risk factors detected"]


def interactive_predict():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run:  python credit_intelligence.py train")
        return
    print("\n--- Credit Intelligence: applicant scoring ---")
    applicant = {key: float(input(f"{label}: ")) for key, label in FEATURES.items()}
    result = score_applicant(applicant)

    print("\n=========== RESULT ===========")
    print(f"Credit score        : {result['credit_score']}")
    print(f"Risk band           : {result['risk_band']}")
    print(f"Default probability : {result['default_probability']:.1%}")
    print(f"Decision            : {result['decision']}")
    print("Key factors:")
    for r in result["key_factors"]:
        print(f"  - {r}")


# ----------------------------------------------------------------- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Intelligence")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train and evaluate the model")
    t.add_argument("--csv", help="Path to your own dataset (optional)")
    sub.add_parser("predict", help="Score a new applicant interactively")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.csv)
    else:
        interactive_predict()
