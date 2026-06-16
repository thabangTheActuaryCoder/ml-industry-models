"""Preprocessing configuration for the mining equipment failure model."""

NUMERIC_FEATURES = [
    "temperature_celsius",
    "vibration_mm_s",
    "oil_pressure_kpa",
    "rpm",
    "operating_hours",
    "days_since_maintenance",
    "load_percentage",
    "ambient_temperature_celsius",
    "hydraulic_pressure_kpa",
    "num_previous_failures",
]

CATEGORICAL_FEATURES = [
    "equipment_type",
    "mine_type",
    "shift",
    "province",
]

TARGET_COLUMN = "target"

TARGET_NAMES = ["Operational", "Failure"]
