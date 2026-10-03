import os
import traceback
import warnings

import numpy as np
import pandas as pd

# Prevent slow wmic queries and loky warnings on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from sklearn.base import BaseEstimator, TransformerMixin, clone
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
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
_RF_TREES_LARGE    =     80   # Trees for RF when n > 2500 (was 50)
_RF_TREES_SMALL    =    150   # Trees for RF on small data (was 100)
_HIST_ITER         =    100   # HistGB iterations (was 60)
_HIST_LEAF         =     31   # HistGB max_leaf_nodes

# ─── Problem-type thresholds ─────────────────────────────────
# A numeric column is treated as classification only when ALL of:
#   (a) it is integer-typed (or boolean), AND
#   (b) the number of unique values is ≤ _CLF_MAX_UNIQUE, AND
#   (c) its unique-value ratio (nunique/nrows) is ≤ _CLF_MAX_RATIO
# Otherwise it is regression.
_CLF_MAX_UNIQUE = 20      # absolute upper bound on class count
_CLF_MAX_RATIO  = 0.05    # at most 5 % distinct values


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
      • object cols with >max_cat unique values → normalized frequency encoding
    """

    def __init__(self, max_cat: int = 50):
        self.max_cat = max_cat

    def fit(self, X, y=None):
        self.drop_cols_: list = []
        self.str_cols_: list = []
        self.bool_cols_: list = []
        self.freq_cols_: dict = {}

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

                # Very-high cardinality text → normalized frequency encoding
                if X[col].nunique(dropna=True) > self.max_cat:
                    counts = X[col].dropna().astype(str).value_counts(normalize=True)
                    self.freq_cols_[col] = counts.to_dict()
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

        # High-cardinality → frequency encoding (continuous feature 0.0 to 1.0)
        for col, freq_map in self.freq_cols_.items():
            if col in X.columns:
                min_freq = min(freq_map.values()) if freq_map else 0.0001
                fallback = min_freq * 0.5
                X[col] = (
                    X[col].astype(str)
                    .map(freq_map)
                    .fillna(fallback)
                    .astype(float)
                )

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
    Uses RobustScaler to better handle outliers.
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
                ("scl", RobustScaler()),   # robust to outliers
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
                    max_categories=50,   # increased from 30
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
                    max_categories=50,
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
    """
    Detect columns that are IDs and should not be used as features.
    Checks name-based heuristics, high-uniqueness strings, and
    sequential integer patterns.
    """
    id_columns = []
    n = len(df)
    id_keywords = {
        "id", "identifier", "uuid", "guid", "key",
        "index", "idx", "no", "num", "number", "code", "ref",
    }

    for col in df.columns:
        col_lower = col.lower().strip()
        col_parts = set(col_lower.replace("-", "_").split("_"))

        # Name-based: any part of the column name matches an id keyword
        name_is_id = bool(col_parts & id_keywords)

        nuniq = df[col].nunique(dropna=True)
        ratio = nuniq / max(n, 1)

        dtype = df[col].dtype
        is_object = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        is_int = pd.api.types.is_integer_dtype(dtype)

        # High-uniqueness string column
        if is_object and ratio >= 0.90:
            id_columns.append(col)
            continue

        # Name matches id keyword AND nearly all unique
        if name_is_id and ratio >= 0.70:
            id_columns.append(col)
            continue

        # Integer column that is sequential (step=1) with near 100% unique
        if is_int and ratio >= 0.98:
            vals = df[col].dropna().sort_values().diff().dropna()
            if len(vals) > 0 and vals.mode().iloc[0] == 1:
                id_columns.append(col)
                continue

    return id_columns


# =============================================================
# Helper: Detect Problem Type
# =============================================================

def detect_problem_type(y: pd.Series, n_rows: int = None):
    """
    Robustly decide between classification and regression.

    Rules (in priority order):
    1. Non-numeric dtype (object/string/category/bool) → classification
    2. Boolean or truly binary (exactly 2 integer values) → classification
    3. Float dtype:
       - If all values are whole numbers AND nunique <= _CLF_MAX_UNIQUE
         AND unique_ratio <= _CLF_MAX_RATIO → classification
       - Otherwise → regression
    4. Integer dtype:
       - If nunique <= _CLF_MAX_UNIQUE AND unique_ratio <= _CLF_MAX_RATIO
         → classification
       - Otherwise → regression
    """
    if n_rows is None:
        n_rows = len(y)

    # Rule 1: Non-numeric
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    # Rule 2: Boolean
    if pd.api.types.is_bool_dtype(y):
        return "classification"

    nuniq = y.nunique(dropna=True)
    ratio = nuniq / max(n_rows, 1)

    # Rule 3: Float
    if pd.api.types.is_float_dtype(y):
        non_null = y.dropna()
        if len(non_null) > 0:
            is_whole = (non_null == non_null.round()).all()
            if is_whole and nuniq <= _CLF_MAX_UNIQUE and ratio <= _CLF_MAX_RATIO:
                return "classification"
        return "regression"

    # Rule 4: Integer
    if nuniq <= _CLF_MAX_UNIQUE and ratio <= _CLF_MAX_RATIO:
        return "classification"

    return "regression"


# =============================================================
# Helper: Detect Target Column
# =============================================================

# Keyword sets for target column name matching
_TARGET_KEYWORDS_HIGH = {
    "target", "label", "class", "output", "prediction",
    "churn", "outcome", "result", "response", "y",
    "survived", "survival", "default", "fraud", "spam",
    "diagnosis", "disease", "status", "approved", "cancelled",
    "converted", "clicked", "purchased", "flagged",
}
_TARGET_KEYWORDS_MEDIUM = {
    "score", "grade", "rating", "price", "salary",
    "revenue", "sales", "value", "amount", "cost",
    "profit", "loss", "return", "yield", "rate",
}


def detect_target_column(df: pd.DataFrame):
    """
    Improved multi-signal target column detection.

    Strategy:
      1. Exact high-confidence keyword match → immediate return.
      2. Partial keyword match (e.g. 'is_churn', 'loan_default').
      3. Score every column across multiple signals:
           - Column name contains a known target keyword
           - Column position (last column = strongest signal)
           - Low cardinality (binary > small categorical > continuous)
           - Non-ID column
           - Non-numeric dtype favoured for classification targets
      4. Return highest-scoring candidate.
      5. Fall back to last non-ID column.
    """
    if df.empty or len(df.columns) == 0:
        return None

    cols = df.columns.tolist()
    lower_map = {c.lower().strip(): c for c in cols}

    # ── Step 1: exact keyword match ───────────────────────────
    for keyword in _TARGET_KEYWORDS_HIGH:
        if keyword in lower_map:
            return lower_map[keyword]

    # ── Step 2: partial keyword match in column parts ─────────
    for col in cols:
        col_parts = set(col.lower().replace("-", "_").split("_"))
        if col_parts & _TARGET_KEYWORDS_HIGH:
            return col

    # ── Step 3: Scoring ───────────────────────────────────────
    id_cols_set = set(detect_id_columns(df))
    scores = {}

    for i, col in enumerate(cols):
        score = 0
        col_parts = set(col.lower().replace("-", "_").split("_"))
        nuniq = df[col].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)

        if col in id_cols_set:
            scores[col] = -999
            continue

        # Position bonus: last column is most likely the target
        if i == len(cols) - 1:
            score += 4
        elif i == len(cols) - 2:
            score += 1

        # Medium keyword match
        if col_parts & _TARGET_KEYWORDS_MEDIUM:
            score += 3

        # Cardinality signals
        if nuniq == 2:
            score += 5      # binary → very likely classification target
        elif 3 <= nuniq <= 10:
            score += 3      # small categorical
        elif 11 <= nuniq <= _CLF_MAX_UNIQUE and not is_numeric:
            score += 1
        elif is_numeric and nuniq > _CLF_MAX_UNIQUE:
            score += 2      # continuous numeric → regression target

        # Non-numeric columns are more likely class labels
        if not is_numeric:
            score += 2

        # Penalise high-cardinality numeric features that are not at the end
        if is_numeric and nuniq > 50 and i < len(cols) - 3:
            score -= 2

        scores[col] = score

    if not scores:
        return None

    best_col = max(scores, key=lambda c: scores[c])
    if scores[best_col] <= 0:
        # Last resort: return last non-ID column
        for col in reversed(cols):
            if col not in id_cols_set:
                return col
        return cols[-1]

    return best_col


# =============================================================
# Classification Metrics
# =============================================================

def calculate_classification_metrics(model, X_test, y_test, y_pred):
    accuracy     = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall       = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1           = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm           = confusion_matrix(y_test, y_pred)

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
        "accuracy":          round(float(accuracy),     4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "precision":         round(float(precision),    4),
        "recall":            round(float(recall),       4),
        "f1_score":          round(float(f1),           4),
        "roc_auc":           round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix":  cm.tolist(),
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
        "r2_score": round(float(r2),   4),
        "mae":      round(float(mae),  4),
        "mse":      round(float(mse),  4),
        "rmse":     round(float(rmse), 4),
    }


# =============================================================
# Fast Hyperparameter Fine-Tuning on Top Model Family
# =============================================================

def _tune_top_model(
    best_name: str,
    best_pipeline,
    X_fit,
    y_fit,
    X_te,
    y_te,
    problem_type: str = "classification",
):
    """
    Fast hyperparameter fine-tuning on the top-performing model family.
    Runs a fast 3-fold cross-validated search (4-6 iterations) on X_fit/y_fit,
    then evaluates on the holdout test set (X_te, y_te).
    Returns {"metrics": metrics, "best_params": search.best_params_, "pipeline": best_est} or None.
    """
    if best_pipeline is None:
        return None

    name_clean = best_name.lower()
    param_dist = {}

    if "gradient boosting" in name_clean or "hist" in name_clean:
        param_dist = {
            "model__learning_rate": [0.05, 0.1, 0.15],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.5, 2.0],
        }
    elif "random forest" in name_clean:
        param_dist = {
            "model__n_estimators": [60, 100, 150],
            "model__max_depth": [8, 15, None],
            "model__min_samples_split": [2, 5],
        }
    elif "logistic" in name_clean:
        param_dist = {
            "model__C": [0.05, 0.2, 1.0, 5.0, 20.0],
        }
    elif "decision tree" in name_clean:
        param_dist = {
            "model__max_depth": [4, 7, 10, 15],
            "model__min_samples_leaf": [2, 5, 10],
        }
    elif "linear svm" in name_clean or "linearsvc" in name_clean:
        param_dist = {
            "model__C": [0.1, 0.5, 1.0, 5.0, 10.0],
        }
    elif "ridge" in name_clean:
        param_dist = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif "lasso" in name_clean or "elasticnet" in name_clean:
        param_dist = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
        }
    elif "neighbor" in name_clean or "knn" in name_clean:
        param_dist = {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
        }

    if not param_dist:
        return None

    try:
        # Cap samples to keep tuning super fast (<= 3000 rows)
        X_tune, y_tune = _sample(
            X_fit, y_fit, cap=3000, stratify=(problem_type == "classification")
        )

        n_combos = 1
        for v in param_dist.values():
            n_combos *= len(v)
        n_iter = min(5, n_combos)

        scoring = "accuracy" if problem_type == "classification" else "r2"
        cv = 3
        if problem_type == "classification":
            min_class_count = pd.Series(y_tune).value_counts().min()
            if min_class_count < 3:
                cv = 2

        search = RandomizedSearchCV(
            estimator=best_pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_tune, y_tune)
        best_est = search.best_estimator_

        y_pred = best_est.predict(X_te)
        if problem_type == "classification":
            metrics = calculate_classification_metrics(best_est, X_te, y_te, y_pred)
        else:
            metrics = calculate_regression_metrics(y_te, y_pred)

        return {
            "metrics": metrics,
            "best_params": search.best_params_,
            "pipeline": best_est,
        }
    except Exception as e:
        print(f"[INFO] Hyperparameter tuning skipped or failed: {e}")
        return None


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

    problem_type = detect_problem_type(y, n_rows=len(y))
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

        # Encode y to ensure numeric labels work with all models
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

        # Detect class imbalance
        class_counts = y_enc.value_counts()
        imbalance_ratio = class_counts.iloc[0] / max(class_counts.iloc[-1], 1)
        is_imbalanced = imbalance_ratio > 1.5
        if is_imbalanced:
            print(f"[INFO] Imbalanced dataset (ratio={imbalance_ratio:.2f}x) — using class_weight='balanced'.")

        # Stratified split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.20, random_state=42, stratify=y_enc,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.20, random_state=42,
            )

        n_tr = len(X_tr)
        is_large = n_tr > 2500

        # ── Model definitions ────────────────────────────────────
        models = {
            "Logistic Regression": (
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    tol=1e-4,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced" if is_imbalanced else None,
                    multi_class="auto",
                    C=1.0,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Random Forest": (
                RandomForestClassifier(
                    n_estimators=_RF_TREES_LARGE if is_large else _RF_TREES_SMALL,
                    max_depth=15 if is_large else None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced" if is_imbalanced else None,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Gradient Boosting (Hist)": (
                HistGradientBoostingClassifier(
                    max_iter=_HIST_ITER,
                    max_leaf_nodes=_HIST_LEAF,
                    early_stopping=True,
                    random_state=42,
                    learning_rate=0.1,
                    l2_regularization=0.1,
                    class_weight="balanced" if is_imbalanced else None,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "Decision Tree": (
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=5,
                    min_samples_split=10,
                    random_state=42,
                    class_weight="balanced" if is_imbalanced else None,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
            "K-Nearest Neighbors": (
                KNeighborsClassifier(
                    n_neighbors=min(7, max(3, int(np.sqrt(n_tr) // 2))),
                    n_jobs=-1,
                    weights="distance",
                ),
                _KNN_CAP,
                False,
            ),
            "Naive Bayes": (
                GaussianNB(),
                _NB_CAP,
                True,
            ),
            "Linear SVM": (
                LinearSVC(
                    max_iter=2000,
                    random_state=42,
                    dual="auto",
                    class_weight="balanced" if is_imbalanced else None,
                    C=1.0,
                ),
                _GLOBAL_SAMPLE_CAP,
                False,
            ),
        }

        results = {}
        fitted_pipelines = {}

        for model_name, (model, cap, is_nb) in models.items():
            print(f"\n{'-'*60}\nTraining: {model_name}")
            pipeline = _make_pipeline(model, nb=is_nb)
            X_fit, y_fit = _sample(X_tr, y_tr, cap, stratify=True)
            try:
                pipeline.fit(X_fit, y_fit)
                fitted_pipelines[model_name] = pipeline
                y_pred = pipeline.predict(X_te)
                metrics = calculate_classification_metrics(
                    pipeline, X_te, y_te, y_pred
                )
                results[model_name] = metrics
                print(
                    f"  Accuracy={metrics['accuracy']}  "
                    f"BalAcc={metrics.get('balanced_accuracy')}  "
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

        # ── Best by Accuracy ───────────────────────────────────
        best_acc = max(v["accuracy"] for v in valid.values())
        best = [m for m, v in valid.items() if v["accuracy"] == best_acc]
        best_model = best[0] if len(best) == 1 else "Tie: " + ", ".join(best)

        # ── Fast hyperparameter fine-tuning on top model family ──
        winning_name = best[0]
        if winning_name in fitted_pipelines:
            print(f"\n{'-'*60}\nTuning Top Model: {winning_name}")
            tuned_res = _tune_top_model(
                best_name=winning_name,
                best_pipeline=fitted_pipelines[winning_name],
                X_fit=X_tr,
                y_fit=y_tr,
                X_te=X_te,
                y_te=y_te,
                problem_type="classification",
            )
            if tuned_res and "metrics" in tuned_res:
                t_metrics = tuned_res["metrics"]
                tuned_label = f"{winning_name} (Tuned)"
                # Keep tuned model if it matches or beats baseline accuracy
                if t_metrics["accuracy"] >= best_acc:
                    valid[tuned_label] = t_metrics
                    if t_metrics["accuracy"] > best_acc:
                        best_acc = t_metrics["accuracy"]
                        best_model = tuned_label
                        print(f"  [SUCCESS] Tuning improved Accuracy to {best_acc:.4f} (params: {tuned_res.get('best_params')})")
                    else:
                        print(f"  [INFO] Tuned model matched baseline Accuracy ({best_acc:.4f})")

        print(f"\n{'='*60}\nBest Model by Accuracy: {best_model} ({best_acc:.4f})\n{'='*60}")

        return {
            "problem_type":       problem_type,
            "target_column":      target_column,
            "removed_id_columns": id_cols,
            "features":           {"numerical": num_cols, "categorical": cat_cols},
            "classes":            le.classes_.tolist(),
            "n_classes":          n_classes,
            "baseline_accuracy":  round(1.0 / max(n_classes, 1), 4),
            "class_distribution": y.value_counts().to_dict(),
            "train_samples":      len(X_tr),
            "test_samples":       len(X_te),
            "models":             valid,
            "failed_models":      {m: v for m, v in results.items() if "error" in v},
            "best_model":         best_model,
        }

    # ==========================================================
    # REGRESSION
    # ==========================================================
    else:

        # Ensure y is numeric for regression
        y_reg = pd.to_numeric(y, errors="coerce")
        valid_mask = y_reg.notna()
        if valid_mask.sum() < len(y_reg) * 0.5:
            raise ValueError(
                f"Target column '{target_column}' has too many non-numeric values "
                "for regression. Consider treating it as classification."
            )
        X    = X[valid_mask]
        y_reg = y_reg[valid_mask]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_reg, test_size=0.20, random_state=42,
        )

        n_tr = len(X_tr)
        is_large = n_tr > 2500

        models = {
            "Linear Regression": (
                LinearRegression(),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Ridge": (
                Ridge(alpha=1.0, random_state=42),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Lasso": (
                Lasso(alpha=0.01, random_state=42, max_iter=5000),
                _GLOBAL_SAMPLE_CAP,
            ),
            "ElasticNet": (
                ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Random Forest": (
                RandomForestRegressor(
                    n_estimators=_RF_TREES_LARGE if is_large else _RF_TREES_SMALL,
                    max_depth=15 if is_large else None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Gradient Boosting (Hist)": (
                HistGradientBoostingRegressor(
                    max_iter=_HIST_ITER,
                    max_leaf_nodes=_HIST_LEAF,
                    early_stopping=True,
                    random_state=42,
                    learning_rate=0.1,
                    l2_regularization=0.1,
                ),
                _GLOBAL_SAMPLE_CAP,
            ),
            "Decision Tree": (
                DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_leaf=5,
                    min_samples_split=10,
                    random_state=42,
                ),
                _GLOBAL_SAMPLE_CAP,
            ),
            "K-Nearest Neighbors": (
                KNeighborsRegressor(
                    n_neighbors=min(7, max(3, int(np.sqrt(n_tr) // 2))),
                    n_jobs=-1,
                    weights="distance",
                ),
                _KNN_CAP,
            ),
            "Linear SVR": (
                LinearSVR(max_iter=5000, random_state=42, C=0.1),
                _GLOBAL_SAMPLE_CAP,
            ),
        }

        results = {}
        fitted_pipelines = {}

        for model_name, (model, cap) in models.items():
            print(f"\n{'-'*60}\nTraining: {model_name}")
            pipeline = _make_pipeline(model)
            X_fit, y_fit = _sample(X_tr, y_tr, cap, stratify=False)
            try:
                pipeline.fit(X_fit, y_fit)
                fitted_pipelines[model_name] = pipeline
                y_pred   = pipeline.predict(X_te)
                metrics  = calculate_regression_metrics(y_te, y_pred)
                results[model_name] = metrics
                print(
                    f"  R²={metrics['r2_score']}  "
                    f"MAE={metrics['mae']}  "
                    f"RMSE={metrics['rmse']}"
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

        # ── Best by R² ─────────────────────────────────────────
        best_r2 = max(v["r2_score"] for v in valid.values())
        best    = [m for m, v in valid.items() if v["r2_score"] == best_r2]
        best_model = best[0] if len(best) == 1 else "Tie: " + ", ".join(best)

        # ── Fast hyperparameter fine-tuning on top regression model ──
        winning_name = best[0]
        if winning_name in fitted_pipelines:
            print(f"\n{'-'*60}\nTuning Top Regression Model: {winning_name}")
            tuned_res = _tune_top_model(
                best_name=winning_name,
                best_pipeline=fitted_pipelines[winning_name],
                X_fit=X_tr,
                y_fit=y_tr,
                X_te=X_te,
                y_te=y_te,
                problem_type="regression",
            )
            if tuned_res and "metrics" in tuned_res:
                t_metrics = tuned_res["metrics"]
                tuned_label = f"{winning_name} (Tuned)"
                if t_metrics["r2_score"] >= best_r2:
                    valid[tuned_label] = t_metrics
                    if t_metrics["r2_score"] > best_r2:
                        best_r2 = t_metrics["r2_score"]
                        best_model = tuned_label
                        print(f"  [SUCCESS] Tuning improved R² to {best_r2:.4f} (params: {tuned_res.get('best_params')})")
                    else:
                        print(f"  [INFO] Tuned model matched baseline R² ({best_r2:.4f})")

        print(f"\n{'='*60}\nBest Model by R²: {best_model} ({best_r2:.4f})\n{'='*60}")

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