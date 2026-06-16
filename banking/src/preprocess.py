"""Preprocessing configuration for the banking credit scoring model."""

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "credit_score",
    "num_late_payments",
    "debt_to_income_ratio",
    "num_open_accounts",
    "months_since_last_delinquency",
]

CATEGORICAL_FEATURES = [
    "education_level",
    "employment_type",
    "province",
]

TARGET_COLUMN = "target"

TARGET_NAMES = ["No Default", "Default"]
