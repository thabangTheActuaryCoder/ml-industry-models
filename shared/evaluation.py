"""Model evaluation utilities: metrics, reports, and feature importance extraction."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(pipeline, X_test, y_test, target_names=None):
    """Evaluate a fitted pipeline and return a metrics dictionary.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with a `predict` and `predict_proba` method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    target_names : list[str], optional
        Human-readable class names for the report.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, classification_report, y_pred, y_proba
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=False
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_dict": report_dict,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    return metrics


def print_evaluation_summary(metrics, model_name="Model"):
    """Print a formatted summary of evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*60}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


def get_feature_names(pipeline):
    """Extract feature names from a fitted pipeline's ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)

    return feature_names


def get_feature_importance(pipeline):
    """Extract feature importances from a fitted pipeline.

    Returns
    -------
    dict mapping feature name to importance value, sorted descending.
    """
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        raise ValueError("Classifier does not expose feature_importances_")

    importance_dict = dict(zip(feature_names, importances))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
