"""Plotting utilities for model evaluation visualisations."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from pathlib import Path


def plot_confusion_matrix(cm, target_names, output_path, title="Confusion Matrix"):
    """Save a confusion matrix heatmap to disk.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.
    target_names : list[str]
        Class labels.
    output_path : Path or str
        File path for the saved plot.
    title : str
        Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {output_path}")


def plot_roc_curve(y_test, y_proba, output_path, title="ROC Curve"):
    """Save a ROC curve plot to disk.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    output_path : Path or str
        File path for the saved plot.
    title : str
        Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve plot saved to {output_path}")


def plot_feature_importance(importance_dict, output_path, top_n=15, title="Feature Importance"):
    """Save a horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_dict : dict
        Mapping of feature name to importance value (sorted descending).
    output_path : Path or str
        File path for the saved plot.
    top_n : int
        Number of top features to show.
    title : str
        Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(importance_dict.items())[:top_n]
    features = [item[0] for item in reversed(items)]
    importances = [item[1] for item in reversed(items)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
    ax.barh(features, importances, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {output_path}")
