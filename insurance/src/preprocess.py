"""Preprocessing configuration for the insurance fraud detection model."""

NUMERIC_FEATURES = [
    "claim_amount",
    "policy_tenure_months",
    "customer_age",
    "num_prior_claims",
    "premium_amount",
    "days_to_report",
    "witness_present",
    "police_report_filed",
    "vehicle_age_years",
]

CATEGORICAL_FEATURES = [
    "incident_type",
    "province",
]

TARGET_COLUMN = "target"

TARGET_NAMES = ["Legitimate", "Fraud"]
