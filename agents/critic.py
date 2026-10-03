from typing import Dict, Any


def critic_node(state: Dict[str, Any]):

    print("\n" + "=" * 60)
    print("CRITIC / VALIDATION AGENT")
    print("=" * 60)

    ml_report = state.get("ml_report", {})
    eda_report = state.get("eda_report", {})
    quality_report = state.get("quality_report", {})
    cleaning_report = state.get("cleaning_report", {})

    findings = []
    warnings = []

    # =========================================================
    # 1. DATA QUALITY
    # =========================================================

    missing_values = quality_report.get("missing_values", {})

    if missing_values:
        findings.append(
            f"Missing values detected before cleaning: {missing_values}"
        )
    else:
        findings.append("No missing values detected.")

    duplicate_rows = quality_report.get("duplicate_rows", 0)

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows} duplicate rows were detected."
        )
    else:
        findings.append("No duplicate rows detected.")

    # =========================================================
    # 2. CLEANING VALIDATION
    # =========================================================

    remaining_missing = cleaning_report.get(
        "remaining_missing_values",
        0
    )

    if remaining_missing == 0:
        findings.append(
            "Data cleaning successfully removed remaining missing values."
        )
    else:
        warnings.append(
            f"{remaining_missing} missing values remain after cleaning."
        )

    # =========================================================
    # 3. DATASET SIZE
    # =========================================================

    dataset_shape = eda_report.get(
        "dataset_shape",
        {}
    )

    rows = dataset_shape.get("rows")

    if rows is not None:

        findings.append(
            f"Dataset contains {rows} rows."
        )

        if rows < 100:

            warnings.append(
                f"Dataset contains only {rows} observations. "
                "Model performance may not generalize well."
            )

    # =========================================================
    # 4. TEST SET SIZE
    # =========================================================

    test_samples = ml_report.get(
        "test_samples"
    )

    train_samples = ml_report.get(
        "train_samples"
    )

    if train_samples is not None:
        findings.append(
            f"Training set contains {train_samples} samples."
        )

    if test_samples is not None:

        findings.append(
            f"Test set contains {test_samples} samples."
        )

        if test_samples < 30:

            warnings.append(
                f"Test set contains only {test_samples} samples. "
                "Evaluation metrics may be unstable."
            )

    # =========================================================
    # 5. MODEL PERFORMANCE
    # =========================================================

    models = ml_report.get(
        "models",
        {}
    )

    if models:

        findings.append(
            f"Evaluated {len(models)} machine learning models."
        )

        accuracies = []
        f1_scores = []
        roc_auc_scores = []

        for model_name, metrics in models.items():

            accuracy = metrics.get("accuracy")
            f1 = metrics.get("f1_score")
            roc_auc = metrics.get("roc_auc")

            if accuracy is not None:
                accuracies.append(accuracy)

            if f1 is not None:
                f1_scores.append(f1)

            if roc_auc is not None:
                roc_auc_scores.append(roc_auc)

        # -----------------------------------------------------
        # Perfect Accuracy
        # -----------------------------------------------------

        if accuracies and all(
            score == 1.0
            for score in accuracies
        ):

            warnings.append(
                "All evaluated models achieved 100% accuracy. "
                "Given the small dataset and test set, this should "
                "not be interpreted as proof of generalization."
            )

        # -----------------------------------------------------
        # Perfect F1
        # -----------------------------------------------------

        if f1_scores and all(
            score == 1.0
            for score in f1_scores
        ):

            warnings.append(
                "All evaluated models achieved perfect F1 scores."
            )

        # -----------------------------------------------------
        # Perfect ROC-AUC
        # -----------------------------------------------------

        if roc_auc_scores and all(
            score == 1.0
            for score in roc_auc_scores
        ):

            warnings.append(
                "All evaluated models achieved perfect ROC-AUC."
            )

    # =========================================================
    # 6. MODEL COMPARISON
    # =========================================================

    best_model = ml_report.get("best_model")
    problem_type = ml_report.get("problem_type", "classification")
    is_clf = "classification" in str(problem_type).lower()
    metric_key = "accuracy" if is_clf else "r2_score"
    metric_lbl = "accuracy" if is_clf else "R² score"

    if best_model:
        findings.append(
            f"Best model by {metric_lbl}: {best_model}."
        )

        # Check whether all models have same score
        if models:
            score_values = [
                metrics.get(metric_key)
                for metrics in models.values()
                if metrics.get(metric_key) is not None
            ]

            if (
                score_values
                and len(set(score_values)) == 1
            ):
                warnings.append(
                    f"All evaluated models have identical {metric_lbl}s. "
                    "The selected best model should therefore not be "
                    "interpreted as meaningfully superior."
                )

    # =========================================================
    # 7. CORRELATION CHECK
    # =========================================================

    strong_correlations = eda_report.get(
        "strong_correlations",
        []
    )

    high_correlations = []

    for correlation in strong_correlations:

        value = correlation.get(
            "correlation"
        )

        if (
            value is not None
            and abs(value) >= 0.90
        ):

            high_correlations.append(
                correlation
            )

    if high_correlations:

        warnings.append(
            f"{len(high_correlations)} feature pairs have "
            "absolute correlation >= 0.90. "
            "These relationships should be investigated for "
            "possible redundancy or synthetic-data effects."
        )

    # =========================================================
    # 8. OUTLIER CHECK
    # =========================================================

    outliers = eda_report.get(
        "outliers",
        {}
    )

    outlier_columns = []

    for column, info in outliers.items():

        count = info.get(
            "count",
            0
        )

        if count > 0:

            outlier_columns.append(
                f"{column}: {count}"
            )

    if outlier_columns:

        findings.append(
            "Potential outliers detected in: "
            + ", ".join(outlier_columns)
        )

    else:

        findings.append(
            "No statistical outliers detected."
        )

    # =========================================================
    # 9. FINAL RECOMMENDATION
    # =========================================================

    if warnings:

        recommendation = (
            "Results should be interpreted cautiously. "
            "The dataset is small, the test set is limited, "
            "and model performance is unusually strong."
        )

    else:

        recommendation = (
            "No major validation issues detected."
        )

    critic_report = {

        "status": "review_completed",

        "findings": findings,

        "warnings": warnings,

        "recommendation": recommendation
    }

    # =========================================================
    # PRINT REPORT
    # =========================================================

    print("\nFindings:")

    for finding in findings:

        print(f"  ✓ {finding}")

    print("\nWarnings:")

    if warnings:

        for warning in warnings:

            print(f"  ⚠ {warning}")

    else:

        print("  ✓ No major warnings")

    print("\nRecommendation:")

    print(
        f"  {recommendation}"
    )

    # =========================================================
    # RETURN UPDATED STATE
    # =========================================================

    return {

        "critic_report": critic_report,

        "completed_steps": (
            state.get(
                "completed_steps",
                []
            )
            + ["critic / validation"]
        )
    }