"""Project-wide constants and configuration."""

import os
from pathlib import Path

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Project root (one level up from shared/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Industry directories
BANKING_DIR = PROJECT_ROOT / "banking"
INSURANCE_DIR = PROJECT_ROOT / "insurance"
RETAIL_DIR = PROJECT_ROOT / "retail"
MINING_DIR = PROJECT_ROOT / "mining"

# Sub-directory names
DATA_SUBDIR = "data"
ARTEFACTS_SUBDIR = "artefacts"
DEPLOY_SUBDIR = "deploy"

# South African provinces
SA_PROVINCES = [
    "Gauteng",
    "Western Cape",
    "KwaZulu-Natal",
    "Eastern Cape",
    "Free State",
    "Limpopo",
    "Mpumalanga",
    "North West",
    "Northern Cape",
]

# SA mining equipment types
SA_MINING_EQUIPMENT = [
    "Haul Truck",
    "Excavator",
    "Drill Rig",
    "Conveyor Belt",
    "Crusher",
    "Load Haul Dumper",
    "Grader",
    "Bulldozer",
]

# Education levels
EDUCATION_LEVELS = ["Matric", "Diploma", "Bachelors", "Honours", "Masters", "Doctorate"]

# Employment types
EMPLOYMENT_TYPES = ["Permanent", "Contract", "Self-employed", "Part-time"]

# Insurance incident types
INCIDENT_TYPES = ["Collision", "Theft", "Fire", "Vandalism", "Weather Damage", "Other"]


def get_industry_paths(industry_dir):
    """Return a dict of standard paths for an industry directory."""
    return {
        "data": industry_dir / DATA_SUBDIR,
        "artefacts": industry_dir / ARTEFACTS_SUBDIR,
        "deploy": industry_dir / DEPLOY_SUBDIR,
    }
