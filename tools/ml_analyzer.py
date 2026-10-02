import os
import traceback
import warnings

import numpy as np
import pandas as pd

# Prevent slow wmic queries and loky warnings on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# =============================================================
# Thresholds (tune here to adjust speed vs accuracy trade-off)
# =============================================================
_GLOBAL_SAMPLE_CAP = 20_000   # Max rows any model trains on
_KNN_CAP           =  5_000   # KNN is O(n) at predict time
_NB_CAP            = 50_000   # Naive Bayes is very fast
_RF_TREES_LARGE    =     50   # Trees for RF when n > 2500
_RF_TREES_SMALL    =    100
_HIST_ITER         =     60   # HistGB iterations
_HIST_LEAF         =     31   # HistGB max_leaf_nodes


# =============================================================
# DataCleaner — universal first pipeline step
# =============================================================

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Runs at fit() + transform() time as the FIRST pipeline step.

    Handles ALL problematic data types before they reach encoders:
      • datetime64 columns          → dropped
      • object cols with datetimes  → cast to str
      • object cols with mixed types→ cast to str
      • all-NaN columns             → dropped
      • constant columns            → dropped
      • bool columns                → cast to int (0/1)
      • object cols with >max_cat unique values → hashed to int buckets
    """

    def __init__(self, max_cat: int = 50):
        self.max_cat = max_cat

    def fit(self, X, y=None):
        self.drop_cols_: list = []
        self.str_cols_: list = []
        self.bool_cols_: list = []
        self.hash_cols_: list = []

        for col in X.columns:
            # ── All-NaN or zero-variance → drop
            if X[col].isna().all():
                self.drop_cols_.append(col)
                continue
            if X[col].nunique(dropna=True) <= 1:
                self.drop_cols_.append(col)
                continue

            # ── datetime64 → drop
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.drop_cols_.append(col)
                continue

            # ── bool → convert to int
            if pd.api.types.is_bool_dtype(X[col]):
                self.bool_cols_.append(col)
                continue

            # ── object dtype inspection
            if X[col].dtype == object:
                non_null = X[col].dropna()
                if len(non_null) == 0:
                    self.drop_cols_.append(col)
                    continue

                sample = non_null.iloc[:500]
                types_seen = {type(v).__name__ for v in sample}

                # Contains datetime objects → cast to str
                if "datetime" in " ".join(types_seen).lower():
                    self.str_cols_.append(col)
                    continue

                # Mixed non-str types (e.g. int + str) → cast to str
                if len(types_seen) > 1:
                    self.str_cols_.append(col)
                    continue

                # Very-high cardinality text → hash to int
                if X[col].nunique(dropna=True) > self.max_cat:
                    self.hash_cols_.append(col)
                    continue

        return self

    def transform(self, X):
        X = X.copy()

        # Bool → int
        for col in self.bool_cols_:
            if col in X.columns:
                X[col] = X[col].astype(int)

        # Mixed/datetime object → str
        for col in self.str_cols_:
            if col in X.columns:
                X[col] = X[col].astype(str)

        # High-cardinality → hash into int buckets
        for col in self.hash_cols_:
            if col in X.columns:
                X[col] = X[col].astype(str).apply(
                    lambda v: hash(v) % 10000
                ).astype(int)

        # Drop
        drop = [c for c in self.drop_cols_ if c in X.columns]
        if drop:
            X = X.drop(columns=drop)

        return X


# =============================================================
# Helper: Build preprocessor (called AFTER DataCleaner)
# =============================================================

def _build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer from the already-cleaned X.
    X must have been passed through DataCleaner.transform() first.
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    transformers = []

    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]),
            num_cols,
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=30,
                )),
            ]),
            cat_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


# =============================================================
# Helper: Build preprocessor for Naive Bayes (MinMax instead)
# =============================================================

def _build_nb_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", MinMaxScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=30,
                )),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# =============================================================
# Helper: Smart sample cap
# =============================================================

def _sample(X, y, cap, stratify=True):
    """Return (X, y) capped at `cap` rows."""
    if len(X) <= cap:
        return X, y
    try:
        X_s, _, y_s, _ = train_test_split(
            X, y, train_size=cap, random_state=42,
            stratify=y if stratify else None,
        )
    except Exception:
        X_s, _, y_s, _ = train_test_split(
            X, y, train_size=cap, random_state=42,
        )
    return X_s, y_s


# =============================================================
# Helper: Detect ID Columns
# =============================================================

def detect_id_columns(df: pd.DataFrame):
    id_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if ratio >= 0.90:
            id_columns.append(col)
    return id_columns


# =============================================================
# Helper: Detect Problem Type
# =============================================================

def detect_problem_type(y: pd.Series):
    if pd.api.types.is_numeric_dtype(y):
        return "classification" if y.nunique() <= 10 else "regression"
    return "classification"


# =============================================================
# Helper: Detect Target Column
# =============================================================

def detect_target_column(df: pd.DataFrame):
    candidates = [
        "target", "label", "class", "output",
        "prediction", "y", "churn", "outcome", "result",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    for col in df.columns:
        if df[col].nunique(dropna=True) == 2:
            return col

    for col in df.columns:
        if 2 <= df[col].nunique(dropna=True) <= 5:
            return col

    return None


# =============================================================
# Classification Metrics
# =============================================================

def calculate_classification_metrics(model, X_test, y_test, y_pred):
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(X_test)
            classes = model.classes_
            if len(classes) == 2:
                roc_auc = roc_auc_score(y_test, proba[:, 1])
            elif len(classes) > 2:
                roc_auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="weighted"
                )
    except Exception:
        roc_auc = None

    return {
        "accuracy":         round(float(accuracy),  4),
        "precision":        round(float(precision), 4),
        "recall":           round(float(recall),    4),
        "f1_score":         round(float(f1),        4),
        "roc_auc":          round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }


# =============================================================
# Regression Metrics
# =============================================================

def calculate_regression_metrics(y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    return {
        "mae":      round(float(mae),  4),
        "mse":      round(float(mse),  4),
        "rmse":     round(float(rmse), 4),
        "r2_score": round(float(r2),   4),
    }


# =============================================================
# Main ML Analyzer
# =============================================================

def analyze_ml(df: pd.DataFrame, target_column=None):
    """
    Automated ML: classification or regression.

    Speed design:
      - Hard sample cap (_GLOBAL_SAMPLE_CAP) applied to every model
      - HistGradientBoosting always used (50-100x faster than vanilla GB)
      - LinearSVC / LinearSVR instead of kernel SVM (O(n) vs O(n²))
      - RandomForest with reduced trees on large data
      - KNN capped separately (expensive at predict time)
      - All models trained sequentially but efficiently

    Robustness design:
      - DataCleaner handles: datetime, bool, mixed types, all-NaN,
        constants, high-cardinality text (hashed), mixed object cols
      - y (target) coerced if mixed-type
      - Fallback non-stratified split when stratify fails
    """

    print("\n" + "=" * 60)
    print("MACHINE LEARNING ANALYZER")
    print("=" * 60)

    # ----------------------------------------------------------
    # Validate
    # ----------------------------------------------------------
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    # ----------------------------------------------------------
    # Detect target
    # ----------------------------------------------------------
    if target_column is None:
        target_column = detect_target_column(df)

    if target_column is None:
        raise ValueError("Could not automatically detect a target column.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    print(f"\nTarget Column: {target_column}")

    # ----------------------------------------------------------
    # Drop rows with missing target
    # ----------------------------------------------------------
    working_df = df.dropna(subset=[target_column]).copy()
    if working_df.empty:
        raise ValueError("No valid rows after removing missing target values.")

    # ----------------------------------------------------------
    # Target y — coerce mixed types to str
    # ----------------------------------------------------------
    y = working_df[target_column]
    if y.dtype == object:
        types_in_y = {type(v).__name__ for v in y.dropna().iloc[:500]}
        if len(types_in_y) > 1:
            print(f"[INFO] Target has mixed types {types_in_y} — coercing to str.")
            y = y.astype(str)

    # ----------------------------------------------------------
    # Features X — drop target + ID columns
    # ----------------------------------------------------------
    X = working_df.drop(columns=[target_column])

    id_cols = detect_id_columns(X)
    if id_cols:
        print(f"\nRemoving ID columns: {id_cols}")
        X = X.drop(columns=id_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    # ----------------------------------------------------------
    # Pre-clean X to build preprocessor schema
    # ----------------------------------------------------------
    _schema_cleaner = DataCleaner().fit(X)
    X_clean         = _schema_cleaner.transform(X)

    problem_type = detect_problem_type(y)
    print(f"\nProblem Type: {problem_type}")

    preprocessor, num_cols, cat_cols = _build_preprocessor(X_clean)

    print(f"\nNumerical Features: {num_cols}")
    print(f"Categorical Features: {cat_cols}")
    print(f"\nSamples: {len(X)}")

    # ----------------------------------------------------------
    # Helper: build standard pipeline
    # ----------------------------------------------------------
    def _make_pipeline(model, nb=False):
        cleaner = DataCleaner()
        if nb:
            prep = _build_nb_preprocessor(X_clean)
            return Pipeline([
                ("cleaner",     cleaner),
                ("preprocessor", prep),
                ("model",        model),
            ])
        return Pipeline([
            ("cleaner",      cleaner),
            ("preprocessor", preprocessor),
            ("model",        model),
        ])

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================
    if problem_type == "classification":

        n_classes = y.nunique()
        print(f"\nClasses: {n_classes}  |  Distribution:\n{y.value_counts().to_dict()}")

        # Stratified split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42,
            )

        n_tr = len(X_tr)
        is_large = n_tr > 2500

        # ── Model definitions (always fast variants) ───────────
        # HistGradientBoosting is the fastest tree ensemble for
        # large data and handles missing values natively.
        # LinearSVC is O(n) vs RBF-SVM's O(n²/n³).
        models = {
            "Logistic Regression": (
                LogisticRegression(
                    solver="saga", max_iter=300, tol=1e-3,
                    random_state=42, n_jobs=-1,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Random Forest": (
                RandomForestClassifier(
                    n_estimators=_RF_TREES_LARGE if is_large else _RF_TREES_SMALL,
                    max_depth=12 if is_large else None,
                    random_state=42, n_jobs=-1,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Gradient Boosting (Hist)": (
                HistGradientBoostingClassifier(
                    max_iter=_HIST_ITER, max_leaf_nodes=_HIST_LEAF,
                    early_stopping=True, random_state=42,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Decision Tree": (
                DecisionTreeClassifier(max_depth=10, random_state=42),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "K-Nearest Neighbors": (
                KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                _KNN_CAP,
                False,
            ),
            "Naive Bayes": (
                GaussianNB(),
                _NB_CAP,
                True,    # needs MinMax pipeline
            ),
            "Linear SVM": (
                LinearSVC(max_iter=1000, random_state=42, dual="auto"),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
        }

        results = {}

        for model_name, (model, cap, is_nb) in models.items():
            print(f"\n{'-'*60}\nTraining: {model_name}")
            pipeline = _make_pipeline(model, nb=is_nb)
            X_fit, y_fit = _sample(X_tr, y_tr, cap, stratify=True)
            try:
                pipeline.fit(X_fit, y_fit)
                y_pred = pipeline.predict(X_te)
                metrics = calculate_classification_metrics(
                    pipeline, X_te, y_te, y_pred
                )
                results[model_name] = metrics
                print(
                    f"  Accuracy={metrics['accuracy']}  "
                    f"F1={metrics['f1_score']}  "
                    f"ROC-AUC={metrics['roc_auc']}"
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                print(traceback.format_exc())
                results[model_name] = {"error": str(e)}

        # ── Filter valid ───────────────────────────────────────
        valid = {m: v for m, v in results.items() if "error" not in v}

        if not valid:
            details = "\n".join(
                f"  {m}: {v['error']}" for m, v in results.items() if "error" in v
            )
            raise RuntimeError(f"All classification models failed:\n{details}")

        # ── Best by F1 ─────────────────────────────────────────
        best_f1 = max(v["f1_score"] for v in valid.values())
        best = [m for m, v in valid.items() if v["f1_score"] == best_f1]
        best_model = best[0] if len(best) == 1 else "Tie: " + ", ".join(best)

        print(f"\n{'='*60}\nBest Model by F1: {best_model}\n{'='*60}")

        return {
            "problem_type":      problem_type,
            "target_column":     target_column,
            "removed_id_columns": id_cols,
            "features":          {"numerical": num_cols, "categorical": cat_cols},
            "classes":           y.unique().tolist(),
            "class_distribution": y.value_counts().to_dict(),
            "train_samples":     len(X_tr),
            "test_samples":      len(X_te),
            "models":            valid,
            "failed_models":     {m: v for m, v in results.items() if "error" in v},
            "best_model":        best_model,
        }

    # ==========================================================
    # REGRESSION
    # ==========================================================
    else:

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42,
        )

        n_tr = len(X_tr)
        is_large = n_tr > 2500

        models = {
            "Linear Regression": (
                LinearRegression(),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Ridge": (
                Ridge(random_state=42),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Lasso": (
                Lasso(random_state=42, max_iter=2000),
                _GLOBAL_SAMPLE_CAP,
            ),
            "ElasticNet": (
                ElasticNet(random_state=42, max_iter=2000),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Random Forest": (
                RandomForestRegressor(
                    n_estimators=_RF_TREES_LARGE if is_large else _RF_TREES_SMALL,
                    max_depth=12 if is_large else None,
                    random_state=42, n_jobs=-1,
                ),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Gradient Boosting (Hist)": (
                HistGradientBoostingRegressor(
                    max_iter=_HIST_ITER, max_leaf_nodes=_HIST_LEAF,
                    early_stopping=True, random_state=42,
                ),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Decision Tree": (
                DecisionTreeRegressor(max_depth=10, random_state=42),
                _GLOBAL_SAMPLE_CAP,
            ),
            "K-Nearest Neighbors": (
                KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                _KNN_CAP,
            ),
            "Linear SVR": (
                LinearSVR(max_iter=2000, random_state=42),
                _GLOBAL_SAMPLE_CAP,
            ),
        }

        results = {}

        for model_name, (model, cap) in models.items():
            print(f"\n{'-'*60}\nTraining: {model_name}")
            pipeline = _make_pipeline(model)
            X_fit, y_fit = _sample(X_tr, y_tr, cap, stratify=False)
            try:
                pipeline.fit(X_fit, y_fit)
                y_pred   = pipeline.predict(X_te)
                metrics  = calculate_regression_metrics(y_te, y_pred)
                results[model_name] = metrics
                print(
                    f"  MAE={metrics['mae']}  "
                    f"RMSE={metrics['rmse']}  "
                    f"R²={metrics['r2_score']}"
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                print(traceback.format_exc())
                results[model_name] = {"error": str(e)}

        valid = {m: v for m, v in results.items() if "error" not in v}

        if not valid:
            details = "\n".join(
                f"  {m}: {v['error']}" for m, v in results.items() if "error" in v
            )
            raise RuntimeError(f"All regression models failed:\n{details}")

        best_r2 = max(v["r2_score"] for v in valid.values())
        best    = [m for m, v in valid.items() if v["r2_score"] == best_r2]
        best_model = best[0] if len(best) == 1 else "Tie: " + ", ".join(best)

        print(f"\n{'='*60}\nBest Model by R²: {best_model}\n{'='*60}")

        return {
            "problem_type":      problem_type,
            "target_column":     target_column,
            "removed_id_columns": id_cols,
            "features":          {"numerical": num_cols, "categorical": cat_cols},
            "train_samples":     len(X_tr),
            "test_samples":      len(X_te),
            "models":            valid,
            "failed_models":     {m: v for m, v in results.items() if "error" in v},
            "best_model":        best_model,
        }