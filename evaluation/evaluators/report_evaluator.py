import json
from pathlib import Path

from agents.reporter import reporter_node


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "report_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_state():

    return {
        "quality_report": {
            "missing_values": {},
            "duplicate_rows": 0,
            "quality_score": 99.0
        },

        "cleaning_report": {
            "remaining_missing_values": 0
        },

        "eda_report": {
            "dataset_shape": {
                "rows": 50,
                "columns": 10
            },
            "numerical_summary": {},
            "categorical_summary": {},
            "correlations": {},
            "strong_correlations": [],
            "outliers": {}
        },

        "ml_report": {
            "problem_type": "classification",
            "target_column": "churn",
            "models": {
                "Logistic Regression": {
                    "accuracy": 0.95,
                    "precision": 0.94,
                    "recall": 0.95,
                    "f1_score": 0.94
                },
                "Random Forest": {
                    "accuracy": 0.96,
                    "precision": 0.95,
                    "recall": 0.96,
                    "f1_score": 0.95
                }
            },
            "best_model": "Random Forest"
        },

        "critic_report": {
            "status": "review_completed",
            "findings": [
                "No major data quality issues detected."
            ],
            "warnings": [],
            "recommendation": "Results should be interpreted cautiously."
        }
    }


def evaluate_report(case, actual):

    required_sections = case["required_sections"]

    report = actual.get("final_report", "")

    available_sections = {
        "executive_summary": "Executive Summary" in report,
        "data_profile": (
            "Dataset Overview" in report
            or "Data Profile" in report
        ),
        "data_quality": "Data Quality" in report,
        "eda": (
            "Exploratory Data Analysis" in report
            or "EDA" in report
        ),
        "ml_results": (
            "Machine Learning Analysis" in report
            or "ML Analysis" in report
        ),
        "conclusion": "Conclusion" in report
    }

    correct = 0

    for section in required_sections:
        if available_sections.get(section, False):
            correct += 1

    return (
        correct / len(required_sections)
    ) * 100


def run_report_evaluation():

    test_cases = load_test_cases()
    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        state = create_state()

        try:

            actual = reporter_node(state)

            score = evaluate_report(
                case,
                actual
            )

            print(
                f"  Required: "
                f"{case['required_sections']}"
            )

            print(
                f"  Score: "
                f"{score:.2f}%"
            )

        except Exception as e:

            score = 0.0

            print(f"  Error: {e}")
            print(f"  Score: {score:.2f}%")

        results.append({
            "case_id": case["case_id"],
            "score": score
        })

    overall_score = (
        sum(result["score"] for result in results)
        / len(results)
    )

    passed = sum(
        result["score"] == 100
        for result in results
    )

    print("\n" + "=" * 60)
    print("REPORT GENERATION EVALUATION")
    print("=" * 60)

    print(f"Test Cases   : {len(results)}")
    print(f"Passed Cases : {passed}")
    print(f"Accuracy     : {overall_score:.2f}%")

    for result in results:

        status = (
            "PASS"
            if result["score"] == 100
            else "FAIL"
        )

        print(
            f"{result['case_id']} : "
            f"{result['score']:.2f}% [{status}]"
        )


if __name__ == "__main__":
    run_report_evaluation()