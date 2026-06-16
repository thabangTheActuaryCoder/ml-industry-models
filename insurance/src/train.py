"""Training script for the insurance claim fraud detection model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from shared.config import RANDOM_SEED, TEST_SIZE, INSURANCE_DIR
from shared.pipelines import build_classifier_pipeline
from shared.evaluation import evaluate_classifier, print_evaluation_summary, get_feature_importance
from shared.model_export import save_model
from shared.plotting import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

from insurance.src.preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_NAMES,
)


def main():
    data_path = INSURANCE_DIR / "data" / "claim_data.csv"
    artefacts_dir = INSURANCE_DIR / "artefacts"

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

    # Build pipeline with sample_weight approach for imbalanced classes
    # GradientBoostingClassifier does not support class_weight directly,
    # so we compute sample weights manually
    print("Building and fitting pipeline...")
    pipeline = build_classifier_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=RANDOM_SEED,
        n_estimators=250,
        max_depth=4,
    )

    # Compute sample weights for imbalanced classes
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_classifier(pipeline, X_test, y_test, target_names=TARGET_NAMES)
    print_evaluation_summary(metrics, model_name="Insurance Claim Fraud Detection")

    # Feature importance
    importance = get_feature_importance(pipeline)

    # Save artefacts
    save_model(
        pipeline,
        output_dir=artefacts_dir,
        model_name="fraud_detection_model",
        metrics=metrics,
        extra_metadata={
            "industry": "insurance",
            "use_case": "claim_fraud_detection",
            "target_column": TARGET_COLUMN,
            "features_numeric": NUMERIC_FEATURES,
            "features_categorical": CATEGORICAL_FEATURES,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_balancing": "sample_weight_balanced",
        },
    )

    # Generate plots
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        target_names=TARGET_NAMES,
        output_path=artefacts_dir / "confusion_matrix.png",
        title="Insurance Fraud Detection - Confusion Matrix",
    )
    plot_roc_curve(
        y_test,
        metrics["y_proba"],
        output_path=artefacts_dir / "roc_curve.png",
        title="Insurance Fraud Detection - ROC Curve",
    )
    plot_feature_importance(
        importance,
        output_path=artefacts_dir / "feature_importance.png",
        title="Insurance Fraud Detection - Feature Importance",
    )

    print("\nTraining complete. Artefacts saved.")


if __name__ == "__main__":
    main()
