"""Preprocessing configuration for the retail customer churn model."""

NUMERIC_FEATURES = [
    "monthly_spend_zar",
    "days_since_last_purchase",
    "num_support_tickets",
    "loyalty_points",
    "account_age_months",
    "num_returns_last_year",
    "avg_order_value_zar",
    "num_orders_last_6m",
    "discount_usage_rate",
]

CATEGORICAL_FEATURES = [
    "membership_tier",
    "preferred_channel",
    "province",
]

TARGET_COLUMN = "target"

TARGET_NAMES = ["Retained", "Churned"]
