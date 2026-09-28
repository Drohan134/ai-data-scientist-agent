import json
import os

# Dataset preprocessing decisions are deterministic and inexpensive locally.
# Set AI_DATA_SCIENTIST_FAST_MODE=0 to opt back into the slower LLM decision.
GEMINI_EVAL_DISABLED = os.getenv("AI_DATA_SCIENTIST_FAST_MODE", "1").lower() not in {
    "0", "false", "no", "off"
}


def _extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def preprocessing_agent(df, profile=None, quality_report=None):

    print("\n" + "=" * 60)
    print("PREPROCESSING AGENT")
    print("=" * 60)

    numerical_features = df.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    missing_values = {
        column: int(df[column].isna().sum())
        for column in df.columns
        if df[column].isna().sum() > 0
    }

    context = {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "missing_values": missing_values,
    }

    # =========================================================
    # MLEval deterministic mode
    # =========================================================

    if GEMINI_EVAL_DISABLED:

        print("\nMLEVAL MODE: Gemini disabled.")
        print("Using deterministic preprocessing decision.")

        # -----------------------------------------------------
        # Imputation
        # -----------------------------------------------------

        imputation_required = bool(missing_values)

        # -----------------------------------------------------
        # Detect skewed numerical features
        # -----------------------------------------------------

        skewed_features = []

        for column in numerical_features:

            series = df[column].dropna()

            if len(series) < 3:
                continue

            unique_values = df[column].nunique()

            skewness = abs(float(series.skew()))

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            extreme_outlier = False

            if iqr > 0:
                upper_limit = q3 + 3 * iqr
                lower_limit = q1 - 3 * iqr

                extreme_outlier = (
                    series.max() > upper_limit
                    or series.min() < lower_limit
                )

            # Detect extreme two-value distributions such as:
            # [1, 1, 1, 1, 100000]

            extreme_magnitude = False

            if unique_values == 2:

                minimum = float(series.min())
                maximum = float(series.max())

                if minimum > 0:
                    magnitude_ratio = maximum / minimum
                    extreme_magnitude = magnitude_ratio > 100

            if (
                skewness > 0.75
                or extreme_outlier
                or extreme_magnitude
            ):
                skewed_features.append(column)

        # -----------------------------------------------------
        # Scaling
        # -----------------------------------------------------

        scaling_features = [
            column
            for column in numerical_features
            if column not in skewed_features
        ]

        typical_values = []

        for column in scaling_features:

            median = float(df[column].median())

            if abs(median) > 0:
                typical_values.append(abs(median))

        scaling = False

        if len(typical_values) >= 2:

            scale_ratio = (
                max(typical_values)
                / min(typical_values)
            )

            scaling = scale_ratio > 100

        # -----------------------------------------------------
        # Encoding
        # -----------------------------------------------------

        encoding_required = bool(categorical_features)

        # -----------------------------------------------------
        # Transformation
        # -----------------------------------------------------

        transformation_required = bool(skewed_features)

        return {
            "imputation": {
                "required": imputation_required,
                "method": "median" if imputation_required else None,
                "reason": (
                    "Missing values detected."
                    if imputation_required
                    else "No missing values."
                ),
            },
            "scaling": {
                "required": scaling,
                "method": "standard" if scaling else None,
                "reason": (
                    "Numerical features have substantially "
                    "different typical scales."
                    if scaling
                    else "Numerical features have comparable scales."
                ),
            },
            "encoding": {
                "required": encoding_required,
                "method": "one_hot" if encoding_required else None,
                "reason": (
                    "Categorical features detected."
                    if encoding_required
                    else "No categorical features."
                ),
            },
            "transformation": {
                "required": transformation_required,
                "method": "log1p" if transformation_required else None,
                "reason": (
                    "Strong skewness or extreme outliers detected."
                    if transformation_required
                    else "No strong skewness detected."
                ),
            },
        }

    # =========================================================
    # LLM mode
    # =========================================================

    from agents.supervisor import get_llm, get_fallback_llm

    prompt = f"""
You are a data preprocessing planning agent.

Analyze the dataset characteristics below and decide what
preprocessing operations are required.

Dataset information:

{json.dumps(context, indent=2, default=str)}

Decide:

1. imputation
2. scaling
3. encoding
4. transformation

Return ONLY valid JSON in this structure:

{{
    "imputation": {{
        "required": true,
        "method": "median",
        "reason": "..."
    }},
    "scaling": {{
        "required": true,
        "method": "standard",
        "reason": "..."
    }},
    "encoding": {{
        "required": true,
        "method": "one_hot",
        "reason": "..."
    }},
    "transformation": {{
        "required": true,
        "method": "log1p",
        "reason": "..."
    }}
}}

Do not invent dataset characteristics.
"""

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    try:

        llm = get_llm()

        response = llm.invoke(prompt)

        content = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        return _extract_json(content)

    except Exception as gemini_error:

        print(
            "\nGemini preprocessing decision failed:"
            f" {gemini_error}"
        )

        # -----------------------------------------------------
        # Groq fallback
        # -----------------------------------------------------

        try:

            fallback_llm = get_fallback_llm()

            response = fallback_llm.invoke(prompt)

            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            return _extract_json(content)

        except Exception as fallback_error:

            print(
                "\nFallback preprocessing decision failed:"
                f" {fallback_error}"
            )

            # -------------------------------------------------
            # Final fallback
            # -------------------------------------------------

            return {
                "imputation": {
                    "required": bool(missing_values),
                    "method": "median" if missing_values else None,
                    "reason": (
                        "Missing values detected."
                        if missing_values
                        else "No missing values."
                    ),
                },
                "scaling": {
                    "required": False,
                    "method": None,
                    "reason": "Fallback decision.",
                },
                "encoding": {
                    "required": bool(categorical_features),
                    "method": (
                        "one_hot"
                        if categorical_features
                        else None
                    ),
                    "reason": (
                        "Categorical features detected."
                        if categorical_features
                        else "No categorical features."
                    ),
                },
                "transformation": {
                    "required": False,
                    "method": None,
                    "reason": "Fallback decision.",
                },
            }