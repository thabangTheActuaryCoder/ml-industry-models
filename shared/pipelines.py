"""Factory functions for sklearn Pipeline and ColumnTransformer construction."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def build_column_transformer(numeric_features, categorical_features):
    """Build a ColumnTransformer that scales numeric and one-hot encodes categorical features."""
    transformers = []

    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_classifier_pipeline(
    numeric_features,
    categorical_features,
    classifier=None,
    **clf_kwargs,
):
    """Build a full sklearn Pipeline: preprocessing + classifier.

    Parameters
    ----------
    numeric_features : list[str]
        Columns to apply StandardScaler to.
    categorical_features : list[str]
        Columns to apply OneHotEncoder to.
    classifier : sklearn estimator, optional
        If None, defaults to GradientBoostingClassifier.
    **clf_kwargs
        Extra keyword arguments forwarded to the default classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = build_column_transformer(numeric_features, categorical_features)

    if classifier is None:
        defaults = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        defaults.update(clf_kwargs)
        classifier = GradientBoostingClassifier(**defaults)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
