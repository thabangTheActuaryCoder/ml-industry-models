"""Deploy Gradio Spaces for all four industry models on Hugging Face Hub.

Creates a Space (repo_type='space', space_sdk='gradio') for each industry,
uploads a generated app.py and requirements.txt, and optionally adds them to
the existing HF Collection.

Prerequisites
-------------
- ``huggingface-hub`` installed (``pip install huggingface-hub``)
- ``HF_TOKEN`` environment variable set, **or** run ``huggingface-cli login``

Usage
-----
    python scripts/deploy_gradio_spaces.py
"""

import os
import textwrap
from huggingface_hub import HfApi, login


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _get_hf_token() -> str:
    """Retrieve token from env var or cached CLI login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # Fall back to the cached token from ``huggingface-cli login``
    from huggingface_hub.utils import get_token

    token = get_token()
    if token:
        return token
    raise RuntimeError(
        "No HF token found. Set the HF_TOKEN env var or run "
        "'huggingface-cli login'."
    )


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
REQUIREMENTS_TXT = textwrap.dedent("""\
    scikit-learn==1.8.0
    pandas>=2.0.0
    joblib>=1.3.0
    huggingface-hub>=0.23.0
""")

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


# ---------------------------------------------------------------------------
# Per-industry configuration
# ---------------------------------------------------------------------------
INDUSTRY_CONFIGS = [
    {
        "industry": "Banking",
        "space_repo": "ThabangTheActuaryCoder/banking-credit-scoring",
        "model_repo": "ThabangTheActuaryCoder/banking-credit-scoring-model",
        "model_filename": "credit_scoring_model.joblib",
        "metadata_filename": "credit_scoring_model_metadata.json",
        "title": "Banking Credit Scoring",
        "description": (
            "Predict credit default risk using a South African banking "
            "credit scoring model. Enter applicant details below."
        ),
        "output_labels": ["No Default", "Default"],
        "numeric_features": [
            ("age", 18, 80, 35, 1),
            ("annual_income", 50000, 2000000, 450000, 10000),
            ("employment_years", 0, 40, 8, 1),
            ("loan_amount", 10000, 1000000, 150000, 5000),
            ("credit_score", 300, 850, 720, 1),
            ("num_late_payments", 0, 20, 1, 1),
            ("debt_to_income_ratio", 0.0, 1.0, 0.33, 0.01),
            ("num_open_accounts", 0, 20, 4, 1),
            ("months_since_last_delinquency", 0, 120, 24, 1),
        ],
        "categorical_features": [
            (
                "education_level",
                ["Matric", "Diploma", "Bachelors", "Honours", "Masters", "Doctorate"],
            ),
            (
                "employment_type",
                ["Permanent", "Contract", "Self-employed", "Part-time"],
            ),
            ("province", SA_PROVINCES),
        ],
    },
    {
        "industry": "Insurance",
        "space_repo": "ThabangTheActuaryCoder/insurance-fraud-detection",
        "model_repo": "ThabangTheActuaryCoder/insurance-fraud-detection-model",
        "model_filename": "fraud_detection_model.joblib",
        "metadata_filename": "fraud_detection_model_metadata.json",
        "title": "Insurance Fraud Detection",
        "description": (
            "Detect potentially fraudulent insurance claims using a South "
            "African fraud detection model. Enter claim details below."
        ),
        "output_labels": ["Legitimate", "Fraud"],
        "numeric_features": [
            ("claim_amount", 1000, 500000, 50000, 1000),
            ("policy_tenure_months", 1, 240, 36, 1),
            ("customer_age", 18, 80, 40, 1),
            ("num_prior_claims", 0, 15, 1, 1),
            ("premium_amount", 500, 50000, 5000, 100),
            ("days_to_report", 0, 90, 7, 1),
            ("witness_present", 0, 1, 0, 1),
            ("police_report_filed", 0, 1, 1, 1),
            ("vehicle_age_years", 0, 25, 5, 1),
        ],
        "categorical_features": [
            (
                "incident_type",
                [
                    "Collision",
                    "Theft",
                    "Fire",
                    "Vandalism",
                    "Weather Damage",
                    "Other",
                ],
            ),
            ("province", SA_PROVINCES),
        ],
    },
    {
        "industry": "Retail",
        "space_repo": "ThabangTheActuaryCoder/retail-customer-churn",
        "model_repo": "ThabangTheActuaryCoder/retail-customer-churn-model",
        "model_filename": "customer_churn_model.joblib",
        "metadata_filename": "customer_churn_model_metadata.json",
        "title": "Retail Customer Churn Prediction",
        "description": (
            "Predict customer churn for a South African retail business. "
            "Enter customer details below."
        ),
        "output_labels": ["Retained", "Churned"],
        "numeric_features": [
            ("monthly_spend_zar", 0, 50000, 2500, 100),
            ("days_since_last_purchase", 0, 365, 30, 1),
            ("num_support_tickets", 0, 20, 2, 1),
            ("loyalty_points", 0, 50000, 5000, 100),
            ("account_age_months", 1, 120, 24, 1),
            ("num_returns_last_year", 0, 20, 1, 1),
            ("avg_order_value_zar", 0, 20000, 500, 50),
            ("num_orders_last_6m", 0, 50, 8, 1),
            ("discount_usage_rate", 0.0, 1.0, 0.3, 0.01),
        ],
        "categorical_features": [
            ("membership_tier", ["Bronze", "Silver", "Gold", "Platinum"]),
            ("preferred_channel", ["Online", "In-store", "Both"]),
            ("province", SA_PROVINCES),
        ],
    },
    {
        "industry": "Mining",
        "space_repo": "ThabangTheActuaryCoder/mining-equipment-failure",
        "model_repo": "ThabangTheActuaryCoder/mining-equipment-failure-model",
        "model_filename": "equipment_failure_model.joblib",
        "metadata_filename": "equipment_failure_model_metadata.json",
        "title": "Mining Equipment Failure Prediction",
        "description": (
            "Predict equipment failure for South African mining operations. "
            "Enter equipment sensor readings and operational data below."
        ),
        "output_labels": ["Operational", "Failure"],
        "numeric_features": [
            ("temperature_celsius", 20, 120, 65, 1),
            ("vibration_mm_s", 0, 50, 10, 0.5),
            ("oil_pressure_kpa", 100, 800, 400, 10),
            ("rpm", 500, 3000, 1500, 50),
            ("operating_hours", 0, 50000, 10000, 100),
            ("days_since_maintenance", 0, 365, 60, 1),
            ("load_percentage", 0, 100, 70, 1),
            ("ambient_temperature_celsius", 10, 50, 25, 1),
            ("hydraulic_pressure_kpa", 500, 3000, 1500, 50),
            ("num_previous_failures", 0, 10, 1, 1),
        ],
        "categorical_features": [
            (
                "equipment_type",
                [
                    "Haul Truck",
                    "Excavator",
                    "Drill Rig",
                    "Conveyor Belt",
                    "Crusher",
                    "Load Haul Dumper",
                    "Grader",
                    "Bulldozer",
                ],
            ),
            ("mine_type", ["Open Pit", "Underground", "Alluvial"]),
            ("shift", ["Day", "Night", "Extended"]),
            ("province", SA_PROVINCES),
        ],
    },
]


# ---------------------------------------------------------------------------
# app.py generator
# ---------------------------------------------------------------------------
def _build_app_py(cfg: dict) -> str:
    """Generate the Gradio app.py content for a given industry config."""

    # Build slider lines
    slider_lines = []
    for name, lo, hi, default, step in cfg["numeric_features"]:
        label = name.replace("_", " ").title()
        slider_lines.append(
            f'    gr.Slider(minimum={lo}, maximum={hi}, value={default}, '
            f'step={step}, label="{label}"),'
        )
    sliders_block = "\n".join(slider_lines)

    # Build dropdown lines
    dropdown_lines = []
    for name, choices in cfg["categorical_features"]:
        label = name.replace("_", " ").title()
        dropdown_lines.append(
            f'    gr.Dropdown(choices={choices}, value="{choices[0]}", '
            f'label="{label}"),'
        )
    dropdowns_block = "\n".join(dropdown_lines)

    # Build feature-name list for the predict function
    all_feature_names = [n for n, *_ in cfg["numeric_features"]] + [
        n for n, _ in cfg["categorical_features"]
    ]
    feature_names_str = ", ".join(f'"{f}"' for f in all_feature_names)

    # Build predict function parameter list
    param_names = [n for n, *_ in cfg["numeric_features"]] + [
        n for n, _ in cfg["categorical_features"]
    ]
    predict_params = ", ".join(param_names)

    # Build values list for DataFrame construction
    values_str = ", ".join(param_names)

    # Output labels
    output_labels = cfg["output_labels"]

    app_py = f'''\
"""Gradio app for {cfg["title"]}.

Loads the sklearn model from Hugging Face Hub and serves predictions
via a Gradio web interface.
"""

import json
import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Load model and metadata from HF Hub
# ---------------------------------------------------------------------------
MODEL_REPO = "{cfg["model_repo"]}"
MODEL_FILE = "{cfg["model_filename"]}"
METADATA_FILE = "{cfg["metadata_filename"]}"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

metadata_path = hf_hub_download(repo_id=MODEL_REPO, filename=METADATA_FILE)
with open(metadata_path) as f:
    metadata = json.load(f)

metrics = metadata["metrics"]
OUTPUT_LABELS = {output_labels}
FEATURE_NAMES = [{feature_names_str}]


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict({predict_params}):
    """Build a DataFrame from inputs, run prediction, return label dict."""
    values = [{values_str}]
    df = pd.DataFrame([dict(zip(FEATURE_NAMES, values))])
    probas = model.predict_proba(df)[0]
    return dict(zip(OUTPUT_LABELS, [float(p) for p in probas]))


# ---------------------------------------------------------------------------
# Metrics summary for description
# ---------------------------------------------------------------------------
metrics_md = (
    f"**Model metrics** -- "
    f"Accuracy: {{metrics['accuracy']:.4f}} | "
    f"Precision: {{metrics['precision']:.4f}} | "
    f"Recall: {{metrics['recall']:.4f}} | "
    f"F1: {{metrics['f1']:.4f}} | "
    f"ROC AUC: {{metrics['roc_auc']:.4f}}"
)

description = (
    "{cfg["description"]}\\n\\n"
    + metrics_md
    + "\\n\\n"
    "Model source: "
    "[{cfg["model_repo"]}]"
    "(https://huggingface.co/{cfg["model_repo"]})"
)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=[
{sliders_block}
{dropdowns_block}
    ],
    outputs=gr.Label(label="Prediction"),
    title="{cfg["title"]}",
    description=description,
    article=(
        "Built with scikit-learn and Gradio. "
        "For educational and demonstration purposes only."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
'''
    return app_py


# ---------------------------------------------------------------------------
# Main deployment routine
# ---------------------------------------------------------------------------
def main():
    token = _get_hf_token()
    login(token=token)
    api = HfApi()

    created_spaces = []

    for cfg in INDUSTRY_CONFIGS:
        space_repo = cfg["space_repo"]
        print(f"\n{'='*60}")
        print(f"  {cfg['industry']} - {cfg['title']}")
        print(f"{'='*60}")

        # 1. Create the Space repo
        print(f"  Creating Space: {space_repo} ...")
        api.create_repo(
            repo_id=space_repo,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"  Space repo ready.")

        # 2. Generate and upload requirements.txt
        print("  Uploading requirements.txt ...")
        api.upload_file(
            path_or_fileobj=REQUIREMENTS_TXT.encode(),
            path_in_repo="requirements.txt",
            repo_id=space_repo,
            repo_type="space",
            commit_message=f"Add requirements.txt for {cfg['industry']} Gradio Space",
        )

        # 3. Generate and upload app.py
        app_py_content = _build_app_py(cfg)
        print("  Uploading app.py ...")
        api.upload_file(
            path_or_fileobj=app_py_content.encode(),
            path_in_repo="app.py",
            repo_id=space_repo,
            repo_type="space",
            commit_message=f"Add Gradio app for {cfg['industry']} model",
        )

        space_url = f"https://huggingface.co/spaces/{space_repo}"
        print(f"  Live at: {space_url}")
        created_spaces.append((cfg["industry"], space_repo, space_url))

    # ----- Summary -----
    print(f"\n{'='*60}")
    print("  All Gradio Spaces deployed successfully")
    print(f"{'='*60}\n")
    for industry, repo, url in created_spaces:
        print(f"  {industry:12s} : {url}")

    # ----- Add to HF Collection (best-effort) -----
    print("\n  Attempting to add Spaces to HF Collection ...")
    try:
        # List existing collections to find the SA ML models one
        collections = api.list_collections(owner="ThabangTheActuaryCoder")
        target_collection = None
        for coll in collections:
            if "south-african" in (coll.title or "").lower() or "industry" in (coll.title or "").lower():
                target_collection = coll
                break

        if target_collection:
            for _, repo, _ in created_spaces:
                try:
                    api.add_collection_item(
                        collection_slug=target_collection.slug,
                        item_id=repo,
                        item_type="space",
                    )
                    print(f"    Added {repo} to collection '{target_collection.title}'")
                except Exception as e:
                    print(f"    Could not add {repo} to collection: {e}")
            print(f"\n  Collection: https://huggingface.co/collections/{target_collection.slug}")
        else:
            print("    No matching collection found - skipping.")
    except Exception as e:
        print(f"    Collection update skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
