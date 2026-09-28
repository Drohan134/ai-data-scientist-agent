# tools/ml_analyzer.py

import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")


# =========================================================
# Helper: Detect ID Columns
# =========================================================

def detect_id_columns(df: pd.DataFrame):
    """
    Detect columns that are likely identifiers.

    A column is treated as an ID mainly when:
    - It is categorical/string-like
    - It has very high uniqueness
    - It is not a meaningful numerical feature
    """

    id_columns = []

    for column in df.columns:

        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        unique_ratio = (
            df[column].nunique(dropna=True)
            / max(len(df), 1)
        )

        if unique_ratio >= 0.90:
            id_columns.append(column)

    return id_columns


# =========================================================
# Helper: Detect Problem Type
# =========================================================

def detect_problem_type(y: pd.Series):
    """
    Detect whether the target represents:

    - binary classification
    - multiclass classification
    - regression
    """

    # Numeric target
    if pd.api.types.is_numeric_dtype(y):

        unique_values = y.nunique()

        # Small number of unique numeric values
        # usually indicates classification
        if unique_values <= 10:

            return "classification"

        return "regression"

    # Non-numeric target
    return "classification"


# =========================================================
# Helper: Detect Target Column
# =========================================================

def detect_target_column(df: pd.DataFrame):
    """
    Automatically detect a likely target column.

    Priority:
    1. Common target names
    2. Binary columns
    3. Low-cardinality columns
    """

    target_candidates = [
        "target",
        "label",
        "class",
        "output",
        "prediction",
        "y",
        "churn",
        "outcome",
        "result",
    ]

    # -----------------------------------------------------
    # 1. Common target names
    # -----------------------------------------------------

    lower_to_original = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in target_candidates:

        if candidate in lower_to_original:

            return lower_to_original[candidate]

    # -----------------------------------------------------
    # 2. Binary columns
    # -----------------------------------------------------

    for column in df.columns:

        unique_values = df[column].nunique(
            dropna=True
        )

        if unique_values == 2:

            return column

    # -----------------------------------------------------
    # 3. Low-cardinality columns
    # -----------------------------------------------------

    for column in df.columns:

        unique_values = df[column].nunique(
            dropna=True
        )

        if 2 <= unique_values <= 5:

            return column

    return None


# =========================================================
# Helper: Build Preprocessor
# =========================================================

def build_preprocessor(
    X: pd.DataFrame,
):
    """
    Build preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    # -----------------------------------------------------
    # Numerical pipeline
    # -----------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # -----------------------------------------------------
    # Categorical pipeline
    # -----------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=30,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # Column transformer
    # -----------------------------------------------------

    transformers = []

    if numerical_features:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features,
    )


# =========================================================
# Classification Metrics
# =========================================================

def calculate_classification_metrics(
    model,
    X_test,
    y_test,
    y_pred,
):
    """
    Calculate robust classification metrics.

    Works with labels such as:

    0 / 1
    Yes / No
    Female / Male
    A / B / C
    """

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # -----------------------------------------------------
    # F1
    # -----------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    roc_auc = None

    try:

        if hasattr(model, "predict_proba"):

            y_probability = model.predict_proba(
                X_test
            )

            classes = model.classes_

            # Binary classification
            if len(classes) == 2:

                roc_auc = roc_auc_score(
                    y_test,
                    y_probability[:, 1],
                )

            # Multiclass classification
            elif len(classes) > 2:

                roc_auc = roc_auc_score(
                    y_test,
                    y_probability,
                    multi_class="ovr",
                    average="weighted",
                )

    except (ValueError, IndexError):

        roc_auc = None

    return {
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "precision": round(
            float(precision),
            4,
        ),
        "recall": round(
            float(recall),
            4,
        ),
        "f1_score": round(
            float(f1),
            4,
        ),
        "roc_auc": (
            round(
                float(roc_auc),
                4,
            )
            if roc_auc is not None
            else None
        ),
        "confusion_matrix": cm.tolist(),
    }


# =========================================================
# Regression Metrics
# =========================================================

def calculate_regression_metrics(
    y_test,
    y_pred,
):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred,
    )

    return {
        "mae": round(
            float(mae),
            4,
        ),
        "mse": round(
            float(mse),
            4,
        ),
        "rmse": round(
            float(rmse),
            4,
        ),
        "r2_score": round(
            float(r2),
            4,
        ),
    }


# =========================================================
# Main ML Analyzer
# =========================================================

def analyze_ml(
    df: pd.DataFrame,
    target_column=None,
):
    """
    Perform automated machine learning analysis.

    Supports:

    - Classification
    - Regression
    - Numerical features
    - Categorical features
    - Missing values
    - ID columns
    """

    print("\n" + "=" * 60)
    print("MACHINE LEARNING ANALYZER")
    print("=" * 60)

    # -----------------------------------------------------
    # Validate dataframe
    # -----------------------------------------------------

    if df is None or df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    # -----------------------------------------------------
    # Detect target
    # -----------------------------------------------------

    if target_column is None:

        target_column = detect_target_column(
            df
        )

    if target_column is None:

        raise ValueError(
            "Could not automatically detect a target column."
        )

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )

    print(
        f"\nTarget Column: {target_column}"
    )

    # -----------------------------------------------------
    # Remove rows where target is missing
    # -----------------------------------------------------

    working_df = df.dropna(
        subset=[target_column]
    ).copy()

    if working_df.empty:

        raise ValueError(
            "No valid rows remain after removing "
            "missing target values."
        )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    y = working_df[
        target_column
    ]

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    X = working_df.drop(
        columns=[target_column]
    )

    # -----------------------------------------------------
    # Detect and remove ID columns
    # -----------------------------------------------------

    detected_id_columns = detect_id_columns(
        X
    )

    if detected_id_columns:

        print(
            "\nRemoving ID columns:"
        )

        for column in detected_id_columns:

            print(
                f"  - {column}"
            )

        X = X.drop(
            columns=detected_id_columns
        )

    # -----------------------------------------------------
    # Check feature availability
    # -----------------------------------------------------

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns remain "
            "after removing the target and ID columns."
        )

    # -----------------------------------------------------
    # Detect problem type
    # -----------------------------------------------------

    problem_type = detect_problem_type(
        y
    )

    print(
        f"\nProblem Type: {problem_type}"
    )

    # -----------------------------------------------------
    # Feature types
    # -----------------------------------------------------

    (
        preprocessor,
        numerical_features,
        categorical_features,
    ) = build_preprocessor(X)

    print(
        "\nNumerical Features:"
    )

    print(
        numerical_features
    )

    print(
        "\nCategorical Features:"
    )

    print(
        categorical_features
    )

    # -----------------------------------------------------
    # Dataset size
    # -----------------------------------------------------

    print(
        f"\nSamples: {len(X)}"
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if problem_type == "classification":

        class_count = y.nunique()

        print(
            f"\nNumber of Classes: {class_count}"
        )

        print(
            "\nClass Distribution:"
        )

        print(
            y.value_counts().to_dict()
        )

        # -------------------------------------------------
        # Stratified split
        # -------------------------------------------------

        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )
            )

        except ValueError:

            # Fallback for extremely small / imbalanced data
            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                )
            )

        # -------------------------------------------------
        # Models
        # -------------------------------------------------

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=80,
                ),
        }

        results = {}

        # -------------------------------------------------
        # Train models
        # -------------------------------------------------

        for model_name, model in models.items():

            print(
                "\n" + "-" * 60
            )

            print(
                f"Training: {model_name}"
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            try:

                pipeline.fit(
                    X_train,
                    y_train,
                )

                y_pred = pipeline.predict(
                    X_test
                )

                metrics = (
                    calculate_classification_metrics(
                        pipeline,
                        X_test,
                        y_test,
                        y_pred,
                    )
                )

                results[
                    model_name
                ] = metrics

                print(
                    f"Accuracy: "
                    f"{metrics['accuracy']}"
                )

                print(
                    f"Precision: "
                    f"{metrics['precision']}"
                )

                print(
                    f"Recall: "
                    f"{metrics['recall']}"
                )

                print(
                    f"F1 Score: "
                    f"{metrics['f1_score']}"
                )

                print(
                    f"ROC-AUC: "
                    f"{metrics['roc_auc']}"
                )

            except Exception as e:

                print(
                    f"Model failed: {e}"
                )

                results[
                    model_name
                ] = {
                    "error": str(e)
                }

        # -------------------------------------------------
        # Remove failed models
        # -------------------------------------------------

        valid_results = {

            model: metrics

            for model, metrics in results.items()

            if "error" not in metrics
        }

        if not valid_results:

            raise RuntimeError(
                "All classification models failed."
            )

        # -------------------------------------------------
        # Best model by F1
        # -------------------------------------------------

        best_f1 = max(
            metrics["f1_score"]
            for metrics
            in valid_results.values()
        )

        best_models = [

            model

            for model, metrics
            in valid_results.items()

            if metrics["f1_score"] == best_f1
        ]

        if len(best_models) == 1:

            best_model = best_models[0]

        else:

            best_model = (
                "Tie: "
                + ", ".join(best_models)
            )

        print(
            "\n" + "=" * 60
        )

        print(
            f"Best Model by F1 Score: "
            f"{best_model}"
        )

        print(
            "=" * 60
        )

        # -------------------------------------------------
        # Final Report
        # -------------------------------------------------

        return {

            "problem_type":
                problem_type,

            "target_column":
                target_column,

            "removed_id_columns":
                detected_id_columns,

            "features": {

                "numerical":
                    numerical_features,

                "categorical":
                    categorical_features,
            },

            "classes":
                y.unique().tolist(),

            "class_distribution":
                y.value_counts()
                .to_dict(),

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test),

            "models":
                valid_results,

            "failed_models": {

                model: metrics

                for model, metrics
                in results.items()

                if "error" in metrics
            },

            "best_model":
                best_model,
        }

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        # -------------------------------------------------
        # Train / Test Split
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
            )
        )

        # -------------------------------------------------
        # Models
        # -------------------------------------------------

        models = {

            "Linear Regression":
                LinearRegression(),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=80,
                ),
        }

        results = {}

        # -------------------------------------------------
        # Train models
        # -------------------------------------------------

        for model_name, model in models.items():

            print(
                "\n" + "-" * 60
            )

            print(
                f"Training: {model_name}"
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            try:

                pipeline.fit(
                    X_train,
                    y_train,
                )

                y_pred = pipeline.predict(
                    X_test
                )

                metrics = (
                    calculate_regression_metrics(
                        y_test,
                        y_pred,
                    )
                )

                results[
                    model_name
                ] = metrics

                print(
                    f"MAE: "
                    f"{metrics['mae']}"
                )

                print(
                    f"RMSE: "
                    f"{metrics['rmse']}"
                )

                print(
                    f"R² Score: "
                    f"{metrics['r2_score']}"
                )

            except Exception as e:

                print(
                    f"Model failed: {e}"
                )

                results[
                    model_name
                ] = {
                    "error": str(e)
                }

        # -------------------------------------------------
        # Remove failed models
        # -------------------------------------------------

        valid_results = {

            model: metrics

            for model, metrics
            in results.items()

            if "error" not in metrics
        }

        if not valid_results:

            raise RuntimeError(
                "All regression models failed."
            )

        # -------------------------------------------------
        # Best model by R²
        # -------------------------------------------------

        best_r2 = max(
            metrics["r2_score"]
            for metrics
            in valid_results.values()
        )

        best_models = [

            model

            for model, metrics
            in valid_results.items()

            if metrics["r2_score"] == best_r2
        ]

        if len(best_models) == 1:

            best_model = best_models[0]

        else:

            best_model = (
                "Tie: "
                + ", ".join(best_models)
            )

        print(
            "\n" + "=" * 60
        )

        print(
            f"Best Model by R² Score: "
            f"{best_model}"
        )

        print(
            "=" * 60
        )

        # -------------------------------------------------
        # Final Report
        # -------------------------------------------------

        return {

            "problem_type":
                problem_type,

            "target_column":
                target_column,

            "removed_id_columns":
                detected_id_columns,

            "features": {

                "numerical":
                    numerical_features,

                "categorical":
                    categorical_features,
            },

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test),

            "models":
                valid_results,

            "failed_models": {

                model: metrics

                for model, metrics
                in results.items()

                if "error" in metrics
            },

            "best_model":
                best_model,
        }