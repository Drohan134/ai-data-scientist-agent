import json
from pathlib import Path

import pandas as pd

from tools.data_quality import analyze_data_quality


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "data_quality_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_dataframe(case):

    data = {
        "customer_id": [
            "C001", "C002", "C003",
            "C004", "C005", "C006"
        ],
        "age": [20, 25, 30, 35, 40, 45],
        "income": [
            30000, 40000, 50000,
            60000, 70000, 80000
        ],
        "city": [
            "Pune", "Mumbai", "Pune",
            "Delhi", "Mumbai", "Pune"
        ],
        "churn": [
            "No", "Yes", "No",
            "Yes", "No", "Yes"
        ]
    }

    return pd.DataFrame(data)


def evaluate_quality(case, actual):

    required_checks = case["required_checks"]

    available_checks = {
        "missing_values": "missing_values" in actual,
        "duplicates": "duplicate_rows" in actual,
        "id_columns": "id_columns" in actual,
        "constant_columns": "constant_columns" in actual,
        "target_candidates": (
            "potential_target_columns" in actual
        ),
        "class_distribution": (
            "class_distributions" in actual
        ),
        "quality_score": "quality_score" in actual
    }

    correct = 0

    for check in required_checks:
        if available_checks.get(check, False):
            correct += 1

    return (
        correct / len(required_checks)
    ) * 100


def run_quality_evaluation():

    test_cases = load_test_cases()
    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        df = create_dataframe(case)

        try:

            actual = analyze_data_quality(df)

            score = evaluate_quality(
                case,
                actual
            )

            print(
                f"  Required: "
                f"{case['required_checks']}"
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
    print("DATA QUALITY EVALUATION")
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
    run_quality_evaluation()    