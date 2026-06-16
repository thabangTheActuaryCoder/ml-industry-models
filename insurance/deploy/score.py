"""Azure ML scoring script for the insurance fraud detection model."""

import os
import json
import joblib
import pandas as pd


def init():
    """Load the model when the container starts."""
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "fraud_detection_model.joblib")
    model = joblib.load(model_path)


def run(raw_data):
    """Score incoming requests.

    Parameters
    ----------
    raw_data : str
        JSON string with a "data" key containing a list of records.

    Returns
    -------
    str
        JSON string with predictions and probabilities.
    """
    try:
        payload = json.loads(raw_data)
        df = pd.DataFrame(payload["data"])
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df)[:, 1].tolist()
        return json.dumps({
            "predictions": predictions,
            "probabilities": probabilities,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
