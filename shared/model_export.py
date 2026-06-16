"""Model serialisation with joblib and JSON metadata sidecars."""

import json
import joblib
from datetime import datetime, timezone
from pathlib import Path


def save_model(pipeline, output_dir, model_name, metrics=None, extra_metadata=None):
    """Save a fitted pipeline to disk with a JSON metadata sidecar.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline to serialise.
    output_dir : Path or str
        Directory to save artefacts into.
    model_name : str
        Base filename (without extension).
    metrics : dict, optional
        Evaluation metrics to include in metadata.
    extra_metadata : dict, optional
        Additional metadata fields.

    Returns
    -------
    dict with 'model_path' and 'metadata_path' keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    metadata_path = output_dir / f"{model_name}_metadata.json"

    # Save model
    joblib.dump(pipeline, model_path)

    # Build metadata
    metadata = {
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_pipeline_steps": [step[0] for step in pipeline.steps],
        "classifier_type": type(pipeline.named_steps["classifier"]).__name__,
        "classifier_params": pipeline.named_steps["classifier"].get_params(),
    }

    if metrics:
        serialisable_metrics = {}
        for k, v in metrics.items():
            if k in ("confusion_matrix",):
                serialisable_metrics[k] = v.tolist()
            elif k in ("y_pred", "y_proba"):
                continue  # Skip large arrays
            elif k == "classification_report":
                serialisable_metrics[k] = v
            elif k == "classification_report_dict":
                serialisable_metrics[k] = v
            else:
                serialisable_metrics[k] = float(v) if hasattr(v, "item") else v
        metadata["metrics"] = serialisable_metrics

    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_model(model_path):
    """Load a pipeline from a joblib file.

    Parameters
    ----------
    model_path : Path or str
        Path to the .joblib file.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_metadata(metadata_path):
    """Load a JSON metadata sidecar.

    Parameters
    ----------
    metadata_path : Path or str
        Path to the _metadata.json file.

    Returns
    -------
    dict
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        return json.load(f)
