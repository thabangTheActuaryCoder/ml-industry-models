"""Training script for the retail customer churn prediction model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from shared.config import RANDOM_SEED, TEST_SIZE, RETAIL_DIR
from shared.pipelines import build_classifier_pipeline
from shared.evaluation import evaluate_classifier, print_evaluation_summary, get_feature_importance
from shared.model_export import save_model
from shared.plotting import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

from retail.src.preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_NAMES,
)


def main():
    data_path = RETAIL_DIR / "data" / "customer_data.csv"
    artefacts_dir = RETAIL_DIR / "artefacts"

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build and fit pipeline
    print("Building and fitting pipeline...")
    pipeline = build_classifier_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=RANDOM_SEED,
    )
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_classifier(pipeline, X_test, y_test, target_names=TARGET_NAMES)
    print_evaluation_summary(metrics, model_name="Retail Customer Churn")

    # Feature importance
    importance = get_feature_importance(pipeline)

    # Save artefacts
    save_model(
        pipeline,
        output_dir=artefacts_dir,
        model_name="customer_churn_model",
        metrics=metrics,
        extra_metadata={
            "industry": "retail",
            "use_case": "customer_churn_prediction",
            "target_column": TARGET_COLUMN,
            "features_numeric": NUMERIC_FEATURES,
            "features_categorical": CATEGORICAL_FEATURES,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    )

    # Generate plots
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        target_names=TARGET_NAMES,
        output_path=artefacts_dir / "confusion_matrix.png",
        title="Retail Customer Churn - Confusion Matrix",
    )
    plot_roc_curve(
        y_test,
        metrics["y_proba"],
        output_path=artefacts_dir / "roc_curve.png",
        title="Retail Customer Churn - ROC Curve",
    )
    plot_feature_importance(
        importance,
        output_path=artefacts_dir / "feature_importance.png",
        title="Retail Customer Churn - Feature Importance",
    )

    print("\nTraining complete. Artefacts saved.")


if __name__ == "__main__":
    main()
