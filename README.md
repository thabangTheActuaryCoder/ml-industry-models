# ML Industry Models

Scikit-learn classification models across four South African industries, built on a shared framework with Databricks notebook pipelines, Azure ML deployment configs, and SQL schema definitions.

## Industries

| Industry | Use Case | Dataset Size | Target Rate | Model |
|---|---|---|---|---|
| Banking | Credit default scoring | 10 000 | ~15% default | GradientBoosting |
| Insurance | Claim fraud detection | 8 000 | ~8% fraud | GradientBoosting (balanced) |
| Retail | Customer churn prediction | 12 000 | ~20% churn | GradientBoosting |
| Mining | Equipment failure prediction | 15 000 | ~10% failure | GradientBoosting |

All datasets are synthetically generated with realistic feature distributions and province-level geography (all nine SA provinces).

## Project Structure

```
.
├── shared/                          # Reusable framework
│   ├── config.py                    # Constants, paths, province lists
│   ├── data_generation.py           # BaseSyntheticDataGenerator ABC
│   ├── pipelines.py                 # sklearn Pipeline + ColumnTransformer factory
│   ├── evaluation.py                # Metrics, classification reports, feature importance
│   ├── model_export.py              # joblib serialisation + JSON metadata sidecar
│   └── plotting.py                  # Confusion matrix, ROC curve, feature importance plots
│
├── banking/
│   ├── src/
│   │   ├── generate_data.py         # CreditScoringDataGenerator
│   │   ├── preprocess.py            # Feature lists and target names
│   │   └── train.py                 # Training script
│   ├── data/                        # credit_data.csv
│   ├── artefacts/                   # .joblib model, metadata JSON, plots
│   └── deploy/                      # Azure ML endpoint, deployment, environment YAML + score.py
│
├── insurance/                       # Same layout as banking
│   └── ...                          # ClaimFraudDataGenerator, fraud_detection_model
│
├── retail/                          # Same layout as banking
│   └── ...                          # CustomerChurnDataGenerator, customer_churn_model
│
├── mining/                          # Same layout as banking
│   └── ...                          # EquipmentFailureDataGenerator, equipment_failure_model
│
├── notebooks/
│   ├── _setup.ipynb                 # Shared setup: sys.path, ON_DATABRICKS, Delta helpers
│   ├── run_all_pipelines.ipynb      # Orchestrator for all 4 industries
│   ├── banking/
│   │   ├── 01_generate_data.ipynb   # Generate synthetic data
│   │   ├── 02_train_model.ipynb     # Train + evaluate + export artefacts
│   │   ├── 03_serve_model.ipynb     # Databricks Model Serving endpoint
│   │   └── 04_deploy_azure_ml.ipynb # Azure ML Managed Online Endpoint
│   ├── insurance/                   # Same notebook sequence
│   ├── retail/                      # Same notebook sequence
│   └── mining/                      # Same notebook sequence
│
├── sql/
│   ├── 00_create_schema.sql         # Unity Catalog: ml_models catalog + 4 schemas
│   ├── 01_banking_credit_data.sql
│   ├── 02_insurance_claim_data.sql
│   ├── 03_retail_customer_data.sql
│   ├── 04_mining_equipment_data.sql
│   └── 05_data_validation.sql
│
├── requirements.txt
├── Makefile
└── .gitignore
```

## Getting Started

### Local development

```bash
# Install dependencies
make install

# Generate data, train all models
make all

# Or run individual industries
make generate-banking
make train-banking
```

### Databricks

1. Import the repo into Databricks Repos.
2. Create a cluster with Databricks Runtime 16.4 LTS ML (which includes `scikit-learn`, `pandas`, `azure-identity`, and other dependencies).
3. Run the SQL scripts under `sql/` to create the Unity Catalog schemas.
4. Open `notebooks/run_all_pipelines.ipynb` and run all cells, or run each industry's notebooks individually (`01` through `04`).

## Pipeline Stages

Each industry follows the same four-stage pipeline:

### 01 - Generate Data

Produces a synthetic CSV using the `BaseSyntheticDataGenerator` pattern. On Databricks, writes to a Delta table under the `ml_models` catalog.

### 02 - Train Model

Builds a `StandardScaler + OneHotEncoder + GradientBoostingClassifier` pipeline via `shared/pipelines.py`, evaluates it (accuracy, precision, recall, F1, ROC AUC), and exports:

- `{model_name}.joblib` - serialised sklearn pipeline
- `{model_name}_metadata.json` - hyperparameters, metrics, feature lists
- `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`

### 03 - Serve Model (Databricks)

Creates a Databricks Model Serving endpoint using the MLflow model registry. Supports staging and production profiles with configurable workload sizes.

### 04 - Deploy to Azure ML

Deploys the model to an Azure ML Managed Online Endpoint using the YAML configs under `{industry}/deploy/`. Supports staging (`-staging` suffix, 1 instance) and production (`-prod` suffix, 2 instances) profiles.

**Requirements:**
- `azure-ai-ml` installed on the cluster (not pre-installed in Databricks ML Runtime)
- Databricks secret scope `azure-ml` with keys: `subscription-id`, `resource-group`, `workspace-name`

## Shared Framework

The `shared/` package provides reusable components across all four industries:

| Module | Purpose |
|---|---|
| `config.py` | Project paths, random seed, SA province/equipment/education constants |
| `data_generation.py` | Abstract base class with `generate()` and `generate_and_save()` |
| `pipelines.py` | `build_classifier_pipeline()` factory returning a fitted sklearn Pipeline |
| `evaluation.py` | `evaluate_classifier()` returning a full metrics dict; feature importance extraction |
| `model_export.py` | `save_model()` with joblib + JSON sidecar; `load_model()` / `load_metadata()` |
| `plotting.py` | Confusion matrix, ROC curve, and feature importance bar chart generators |

## Azure ML Deploy Configs

Each industry includes ready-to-use Azure ML deployment files under `{industry}/deploy/`:

- `endpoint.yml` - Managed Online Endpoint definition (key auth)
- `deployment.yml` - Deployment spec (Standard_DS2_v2, scoring script, conda env)
- `environment.yml` - Conda environment with scikit-learn, pandas, numpy, joblib
- `score.py` - Scoring script expecting `{"data": [...]}` JSON payload

## Environment Parameterisation

Notebooks `03` and `04` support staging and production profiles, resolved in this order:

1. Databricks widget `ENVIRONMENT`
2. Environment variable `ENVIRONMENT`
3. Default: `staging`

## Requirements

Core dependencies (see `requirements.txt`):

- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- joblib >= 1.3.0
- azure-ai-ml >= 1.23.0
- azure-identity >= 1.21.0

## Licence

Private repository. All rights reserved.
