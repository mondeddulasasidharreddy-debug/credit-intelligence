# Credit Intelligence

A simple credit risk scoring project. It predicts the probability of default,
converts it to a 300-850 credit score, assigns a risk band and decision, and
lists key risk factors.

## Setup
    pip install -r requirements.txt

## Usage
    python credit_intelligence.py train                # train on synthetic data
    python credit_intelligence.py train --csv data.csv # train on your own data
    python credit_intelligence.py predict              # score a new applicant

## Custom dataset
CSV columns required: age, annual_income, loan_amount, existing_debt,
credit_history_years, late_payments, credit_utilization, employment_years,
and a 0/1 target column named `default`.

## Score bands
| Score   | Risk band    | Decision      |
|---------|--------------|---------------|
| 750+    | Excellent    | Approve       |
| 680-749 | Good         | Approve       |
| 600-679 | Fair         | Manual review |
| 500-599 | Poor         | Decline       |
| <500    | Very poor    | Decline       |
