import os
import re
from typing import Dict, Any

import markdown
from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Ollama model config (inherits env vars set in supervisor.py / .env)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------

def get_ollama_llm():
    """Primary LLM: local Qwen2.5-Coder via Ollama (offline, no API key)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=2048,  # reports need a longer output
    )


def get_llm():
    """Cloud fallback: Gemini API."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from gemini_guard import install_gemini_circuit_breaker

    install_gemini_circuit_breaker()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )


# ---------------------------------------------------------
# Deterministic fallback report
# ---------------------------------------------------------

def generate_fallback_report(
    profile: Dict[str, Any],
    quality_report: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    eda_report: Dict[str, Any],
    visualization_report: Dict[str, Any],
    ml_report: Dict[str, Any],
    critic_report: Dict[str, Any],
) -> str:

    lines = []

    lines.append("# AI Data Scientist Report")
    lines.append("")
    lines.append(
        "> This report was generated deterministically from the analysis results. "
        "Set AI_DATA_SCIENTIST_LLM_REPORT=1 to request an LLM-written report."
    )
    lines.append("")

    # -----------------------------------------------------
    # 1. Executive Summary
    # -----------------------------------------------------

    lines.append("## 1. Executive Summary")
    lines.append("")

    rows = profile.get("rows", "N/A")
    columns = profile.get("columns", "N/A")

    lines.append(
        f"The dataset contains **{rows} rows** and **{columns} columns**."
    )

    quality_score = quality_report.get("quality_score")

    if quality_score is not None:
        lines.append(
            f"The calculated data quality score is **{quality_score}/100**."
        )

    best_model = ml_report.get("best_model")

    if best_model:
        lines.append(
            f"The machine learning analysis evaluated multiple models. "
            f"The reported best-model result was **{best_model}**."
        )

    lines.append(
        "The critic stage identified limitations that should be considered "
        "when interpreting the machine learning results."
    )

    # -----------------------------------------------------
    # 2. Dataset Overview
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 2. Dataset Overview")
    lines.append("")

    lines.append(f"- Rows: {rows}")
    lines.append(f"- Columns: {columns}")

    columns_list = profile.get("columns_list", [])

    if columns_list:
        lines.append("- Columns:")
        for column in columns_list:
            lines.append(f"  - `{column}`")

    missing_values = profile.get("missing_values", {})

    if missing_values:
        lines.append("")
        lines.append("Missing values identified during profiling:")

        for column, count in missing_values.items():
            lines.append(f"- `{column}`: {count}")

    else:
        lines.append("")
        lines.append("No missing values were reported during profiling.")

    # -----------------------------------------------------
    # 3. Data Quality
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 3. Data Quality")
    lines.append("")

    duplicate_rows = quality_report.get("duplicate_rows")

    if duplicate_rows is not None:
        lines.append(f"- Duplicate rows: {duplicate_rows}")

    id_columns = quality_report.get("id_columns", [])

    if id_columns:
        lines.append(
            f"- Potential identifier columns: {', '.join(id_columns)}"
        )

    constant_columns = quality_report.get("constant_columns", [])

    if constant_columns:
        lines.append(
            f"- Constant columns: {', '.join(constant_columns)}"
        )
    else:
        lines.append("- Constant columns: None reported")

    high_cardinality_columns = quality_report.get(
        "high_cardinality_columns",
        []
    )

    if high_cardinality_columns:
        lines.append(
            "- High-cardinality columns: "
            + ", ".join(high_cardinality_columns)
        )

    binary_columns = quality_report.get("binary_columns", [])

    if binary_columns:
        lines.append(
            "- Binary columns: " + ", ".join(binary_columns)
        )

    potential_targets = quality_report.get(
        "potential_target_columns",
        []
    )

    if potential_targets:
        lines.append(
            "- Potential target columns: "
            + ", ".join(potential_targets)
        )

    # -----------------------------------------------------
    # 4. Data Cleaning
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 4. Data Cleaning")
    lines.append("")

    rows_before = cleaning_report.get("rows_before")
    rows_after = cleaning_report.get("rows_after")

    columns_before = cleaning_report.get("columns_before")
    columns_after = cleaning_report.get("columns_after")

    if rows_before is not None and rows_after is not None:
        lines.append(
            f"- Rows: {rows_before} → {rows_after}"
        )

    if columns_before is not None and columns_after is not None:
        lines.append(
            f"- Columns: {columns_before} → {columns_after}"
        )

    missing_before = cleaning_report.get("missing_values_before")

    if missing_before:
        lines.append("")
        lines.append("Missing values before cleaning:")

        for column, count in missing_before.items():
            if count:
                lines.append(f"- `{column}`: {count}")

    missing_after = cleaning_report.get("missing_values_after")

    if missing_after is not None:
        remaining_missing = {
            column: count
            for column, count in missing_after.items()
            if count
        }

        if remaining_missing:
            lines.append("")
            lines.append("Remaining missing values:")

            for column, count in remaining_missing.items():
                lines.append(f"- `{column}`: {count}")
        else:
            lines.append(
                "- Remaining missing values after cleaning: 0"
            )

    # -----------------------------------------------------
    # 5. Exploratory Data Analysis
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 5. Exploratory Data Analysis")
    lines.append("")

    strong_correlations = eda_report.get(
        "strong_correlations",
        []
    )

    if strong_correlations:
        lines.append("Strong observed correlations:")

        for item in strong_correlations:
            if isinstance(item, dict):
                feature_1 = item.get(
                    "feature_1",
                    item.get("feature1", "")
                )

                feature_2 = item.get(
                    "feature_2",
                    item.get("feature2", "")
                )

                correlation = item.get(
                    "correlation",
                    item.get("value", "")
                )

                lines.append(
                    f"- `{feature_1}` ↔ `{feature_2}`: "
                    f"{correlation}"
                )

            else:
                lines.append(f"- {item}")

        lines.append("")
        lines.append(
            "These are observed statistical relationships and should not "
            "be interpreted as evidence of causation."
        )

    else:
        lines.append(
            "No strong correlations were reported by the EDA stage."
        )

    outliers = eda_report.get("outliers", {})

    if outliers:
        lines.append("")
        lines.append("Detected outliers:")

        found_outlier = False

        for column, value in outliers.items():

            if isinstance(value, dict):
                count = value.get(
                    "count",
                    value.get("outlier_count", 0)
                )
            else:
                count = value

            if count:
                found_outlier = True
                lines.append(
                    f"- `{column}`: {count}"
                )

        if not found_outlier:
            lines.append("- No outliers were reported.")

    # -----------------------------------------------------
    # 6. Machine Learning Analysis
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 6. Machine Learning Analysis")
    lines.append("")

    target_column = ml_report.get("target_column")

    if target_column:
        lines.append(
            f"- Target column: `{target_column}`"
        )

    problem_type = ml_report.get("problem_type")

    if problem_type:
        lines.append(
            f"- Problem type: `{problem_type}`"
        )

    train_samples = ml_report.get("train_samples")

    if train_samples is not None:
        lines.append(
            f"- Training samples: {train_samples}"
        )

    test_samples = ml_report.get("test_samples")

    if test_samples is not None:
        lines.append(
            f"- Test samples: {test_samples}"
        )

    removed_ids = ml_report.get(
        "removed_id_columns",
        []
    )

    if removed_ids:
        lines.append(
            "- Removed identifier columns: "
            + ", ".join(removed_ids)
        )

    models = ml_report.get("models", {})

    if models:
        lines.append("")
        lines.append("### Model Results")
        lines.append("")

        for model_name, metrics in models.items():

            lines.append(f"**{model_name}**")

            if isinstance(metrics, dict):

                for metric_name, metric_value in metrics.items():

                    if metric_name == "confusion_matrix":
                        continue

                    if isinstance(metric_value, float):
                        lines.append(
                            f"- {metric_name}: "
                            f"{metric_value:.4f}"
                        )
                    else:
                        lines.append(
                            f"- {metric_name}: "
                            f"{metric_value}"
                        )

            lines.append("")

    if best_model:
        lines.append(
            f"Reported best model: **{best_model}**"
        )

    # -----------------------------------------------------
    # 7. Validation and Critic Findings
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 7. Validation and Critic Findings")
    lines.append("")

    findings = critic_report.get("findings", [])

    if findings:
        lines.append("### Findings")

        for finding in findings:
            lines.append(f"- {finding}")

    warnings = critic_report.get("warnings", [])

    if warnings:
        lines.append("")
        lines.append("### Warnings")

        for warning in warnings:
            lines.append(f"- {warning}")

    recommendation = critic_report.get("recommendation")

    if recommendation:
        lines.append("")
        lines.append("### Critic Recommendation")
        lines.append("")
        lines.append(str(recommendation))

    # -----------------------------------------------------
    # 8. Key Findings
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 8. Key Findings")
    lines.append("")

    if missing_values:
        lines.append(
            "1. Missing data was identified during the profiling stage."
        )
    else:
        lines.append(
            "1. No missing data was reported during profiling."
        )

    if strong_correlations:
        lines.append(
            "2. Several strong statistical relationships were identified "
            "during EDA."
        )
    else:
        lines.append(
            "2. No strong correlations were reported."
        )

    if models:
        lines.append(
            "3. Multiple machine learning models were evaluated."
        )

    if warnings:
        lines.append(
            "4. The critic stage identified limitations that should be "
            "considered before interpreting the model results."
        )

    # -----------------------------------------------------
    # 9. Recommendations
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 9. Recommendations")
    lines.append("")

    lines.append(
        "- Treat the reported model metrics as evaluation results on the "
        "available dataset rather than proof of generalization."
    )

    if test_samples is not None:
        lines.append(
            "- Consider additional validation data or cross-validation "
            "before drawing stronger conclusions from model performance."
        )

    if strong_correlations:
        lines.append(
            "- Investigate the strong feature relationships further to "
            "understand whether they reflect meaningful domain patterns "
            "or dataset construction."
        )

    lines.append(
        "- Review the critic warnings before using the model for "
        "real-world decision making."
    )

    # -----------------------------------------------------
    # 10. Conclusion
    # -----------------------------------------------------

    lines.append("")
    lines.append("## 10. Conclusion")
    lines.append("")

    lines.append(
        "The AI Data Scientist pipeline completed profiling, data quality "
        "analysis, cleaning, EDA, visualization, machine learning analysis, "
        "and validation. The results should be interpreted in the context "
        "of the dataset characteristics and the limitations identified by "
        "the critic stage."
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# Convert Markdown text to PDF elements
# ---------------------------------------------------------

def markdown_to_pdf_elements(report: str, styles):

    elements = []
    lines = report.split("\n")

    paragraph_buffer = []

    def flush_paragraph():

        if paragraph_buffer:

            text = " ".join(
                line.strip()
                for line in paragraph_buffer
                if line.strip()
            )

            if text:

                text = escape_pdf_text(text)

                elements.append(
                    Paragraph(
                        text,
                        styles["BodyCustom"]
                    )
                )

                elements.append(
                    Spacer(1, 8)
                )

            paragraph_buffer.clear()

    in_code_block = False
    code_lines = []

    for line in lines:

        stripped = line.strip()

        # Code block
        if stripped.startswith("```"):

            if in_code_block:

                if code_lines:

                    elements.append(
                        Preformatted(
                            "\n".join(code_lines),
                            styles["CodeCustom"]
                        )
                    )

                    elements.append(
                        Spacer(1, 8)
                    )

                code_lines = []
                in_code_block = False

            else:

                flush_paragraph()
                in_code_block = True

            continue

        if in_code_block:

            code_lines.append(line)
            continue

        # Empty line
        if not stripped:

            flush_paragraph()
            continue

        # H1
        if stripped.startswith("# "):

            flush_paragraph()

            heading = stripped[2:].strip()

            elements.append(
                Paragraph(
                    escape_pdf_text(heading),
                    styles["TitleCustom"]
                )
            )

            elements.append(
                Spacer(1, 12)
            )

        # H2
        elif stripped.startswith("## "):

            flush_paragraph()

            heading = stripped[3:].strip()

            elements.append(
                Paragraph(
                    escape_pdf_text(heading),
                    styles["HeadingCustom"]
                )
            )

            elements.append(
                Spacer(1, 8)
            )

        # H3
        elif stripped.startswith("### "):

            flush_paragraph()

            heading = stripped[4:].strip()

            elements.append(
                Paragraph(
                    escape_pdf_text(heading),
                    styles["SubHeadingCustom"]
                )
            )

            elements.append(
                Spacer(1, 6)
            )

        # Bullet point
        elif stripped.startswith("- "):

            flush_paragraph()

            bullet = stripped[2:].strip()

            elements.append(
                Paragraph(
                    "• " + escape_pdf_text(bullet),
                    styles["BulletCustom"]
                )
            )

            elements.append(
                Spacer(1, 4)
            )

        # Numbered list
        elif re.match(r"^\d+\.\s", stripped):

            flush_paragraph()

            elements.append(
                Paragraph(
                    escape_pdf_text(stripped),
                    styles["BulletCustom"]
                )
            )

            elements.append(
                Spacer(1, 4)
            )

        else:

            paragraph_buffer.append(stripped)

    if in_code_block and code_lines:

        elements.append(
            Preformatted(
                "\n".join(code_lines),
                styles["CodeCustom"]
            )
        )

    flush_paragraph()

    return elements


# ---------------------------------------------------------
# Escape text for ReportLab Paragraph
# ---------------------------------------------------------

def escape_pdf_text(text: str) -> str:

    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Basic Markdown formatting

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"<b>\1</b>",
        text
    )

    text = re.sub(
        r"\*(.*?)\*",
        r"<i>\1</i>",
        text
    )

    return text


# ---------------------------------------------------------
# Generate PDF
# ---------------------------------------------------------

def generate_pdf(report: str, pdf_path: str):

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=15,
        )
    )

    styles.add(
        ParagraphStyle(
            name="HeadingCustom",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor("#1565C0"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubHeadingCustom",
            parent=styles["Heading3"],
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BulletCustom",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            leftIndent=15,
            firstLineIndent=-8,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CodeCustom",
            parent=styles["Code"],
            fontSize=8,
            leading=10,
        )
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
        title="AI Data Scientist Report",
        author="AI Data Scientist Agent",
    )

    elements = markdown_to_pdf_elements(
        report,
        styles
    )

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Generated automatically by AI Data Scientist Agent",
            ParagraphStyle(
                "Footer",
                parent=styles["BodyText"],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.grey,
            )
        )
    )

    doc.build(elements)


# ---------------------------------------------------------
# Reporter Agent
# ---------------------------------------------------------

def reporter_node(state: Dict[str, Any]):

    print("\n" + "=" * 60)
    print("REPORTER AGENT")
    print("=" * 60)

    # -----------------------------------------------------
    # Get analysis results
    # -----------------------------------------------------

    profile = state.get("profile", {})
    quality_report = state.get("quality_report", {})
    cleaning_report = state.get("cleaning_report", {})
    eda_report = state.get("eda_report", {})

    visualization_report = state.get(
        "visualization_report",
        {}
    )

    ml_report = state.get("ml_report", {})
    critic_report = state.get("critic_report", {})

    # -----------------------------------------------------
    # Combine reports
    # -----------------------------------------------------

    report_context = {
        "profile": profile,
        "quality_report": quality_report,
        "cleaning_report": cleaning_report,
        "eda_report": eda_report,
        "visualization_report": visualization_report,
        "ml_report": ml_report,
        "critic_report": critic_report,
    }

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are the final reporting agent in an AI Data Scientist system.

Your job is to generate a professional data science report using
ONLY the information provided below.

IMPORTANT RULES:

1. Do not invent statistics, metrics, findings, or conclusions.

2. Do not create numbers that are not present in the input.

3. Clearly mention limitations identified by the critic agent.

4. Do not claim that a model generalizes well merely because its
evaluation metrics are high.

5. If multiple models have identical performance, report them as
a tie rather than selecting an arbitrary winner.

6. Distinguish observed findings from recommendations.

7. Do not make causal claims from correlations.

8. If the dataset is small, explicitly mention that limitation.

9. Use evidence-based and professional wording.

10. Avoid exaggerated language such as "perfect model",
"guaranteed", "excellent generalization", or "proven cause".

11. If strong correlations are present, describe them as
observed relationships and mention that further investigation
may be required.

Generate the report using this structure:

# AI Data Scientist Report

## 1. Executive Summary

## 2. Dataset Overview

## 3. Data Quality

## 4. Data Cleaning

## 5. Exploratory Data Analysis

## 6. Machine Learning Analysis

## 7. Validation and Critic Findings

## 8. Key Findings

## 9. Recommendations

## 10. Conclusion

Here is the analysis data:

{report_context}
"""

    # -----------------------------------------------------
    # Generate report with Gemini
    # -----------------------------------------------------

    report_source = "ollama"

    try:

        if os.getenv("AI_DATA_SCIENTIST_LLM_REPORT", "0").lower() not in {
            "1", "true", "yes", "on"
        }:
            raise RuntimeError("LLM report generation is disabled (AI_DATA_SCIENTIST_LLM_REPORT != 1). Using deterministic fallback.")

        # -------------------------------------------------------
        # Try Ollama first (offline, fastest)
        # -------------------------------------------------------
        try:
            import time
            print(f"\nGenerating report with Ollama ({OLLAMA_MODEL})...")
            t0 = time.time()
            llm = get_ollama_llm()
            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                report = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                )
            else:
                report = response.content

            if not report or not report.strip():
                raise ValueError("Ollama returned an empty report.")

            elapsed = round(time.time() - t0, 2)
            print(f"\nReport generated with Ollama in {elapsed}s.")
            report_source = "ollama"

        except Exception as ollama_err:
            print(f"\nOllama reporter unavailable: {ollama_err}")
            print("Falling back to Gemini...")

            # ---------------------------------------------------
            # Fallback to Gemini cloud
            # ---------------------------------------------------
            llm = get_llm()
            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                report = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                )
            else:
                report = response.content

            if not report or not report.strip():
                raise ValueError("Gemini returned an empty report.")

            print("\nReport generated successfully using Gemini.")
            report_source = "gemini"

    except Exception as e:

        error_message = str(e)

        print("\nWARNING: Gemini reporter unavailable.")
        print(f"Reason: {error_message}")

        print(
            "\nGenerating deterministic fallback report..."
        )

        report = generate_fallback_report(
            profile,
            quality_report,
            cleaning_report,
            eda_report,
            visualization_report,
            ml_report,
            critic_report,
        )

        report_source = "deterministic_fallback"

        print(
            "Fallback report generated successfully."
        )

    # -----------------------------------------------------
    # Create reports directory
    # -----------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save Markdown
    # -----------------------------------------------------

    report_path = "reports/final_report.md"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print(
        f"Markdown report saved to: {report_path}"
    )

    # -----------------------------------------------------
    # Markdown → HTML
    # -----------------------------------------------------

    html_body = markdown.markdown(
        report,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
        ],
    )

    html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AI Data Scientist Report</title>

<style>

body {{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    max-width: 1000px;

    margin: 40px auto;

    padding: 0 25px;

    line-height: 1.7;

    color: #222;

    background: #f7f9fc;
}}

.container {{
    background: white;

    padding: 40px;

    border-radius: 12px;

    box-shadow:
        0 4px 20px rgba(0,0,0,0.08);
}}

h1 {{
    color: #1565c0;

    border-bottom:
        3px solid #1565c0;

    padding-bottom: 10px;
}}

h2 {{
    color: #1976d2;

    margin-top: 35px;

    border-bottom:
        1px solid #e0e0e0;

    padding-bottom: 6px;
}}

h3 {{
    color: #333;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    margin: 20px 0;
}}

th,
td {{
    border: 1px solid #ddd;

    padding: 10px;

    text-align: left;
}}

th {{
    background: #1976d2;

    color: white;
}}

tr:nth-child(even) {{
    background: #f2f6fa;
}}

code {{
    background: #f1f1f1;

    padding: 3px 6px;

    border-radius: 4px;
}}

pre {{
    background: #f4f4f4;

    padding: 15px;

    border-radius: 8px;

    overflow-x: auto;
}}

blockquote {{
    border-left:
        4px solid #1976d2;

    padding-left: 15px;

    color: #555;
}}

li {{
    margin: 6px 0;
}}

.footer {{
    margin-top: 40px;

    padding-top: 15px;

    border-top:
        1px solid #ddd;

    color: #777;

    font-size: 14px;

    text-align: center;
}}

</style>

</head>

<body>

<div class="container">

{html_body}

<div class="footer">

Generated automatically by
AI Data Scientist Agent

</div>

</div>

</body>

</html>
"""

    # -----------------------------------------------------
    # Save HTML
    # -----------------------------------------------------

    html_path = "reports/final_report.html"

    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html_content)

    print(
        f"HTML report saved to: {html_path}"
    )

    # -----------------------------------------------------
    # Generate PDF
    # -----------------------------------------------------

    pdf_path = "reports/final_report.pdf"

    generate_pdf(
        report,
        pdf_path
    )

    print(
        f"PDF report saved to: {pdf_path}"
    )

    # -----------------------------------------------------
    # Return state
    # -----------------------------------------------------

    return {

        "final_report": report,

        "report_path": report_path,

        "html_report_path": html_path,

        "pdf_report_path": pdf_path,

        "report_source": report_source,

        "completed_steps":
            state.get(
                "completed_steps",
                []
            )
            + ["report generation"],
    }
