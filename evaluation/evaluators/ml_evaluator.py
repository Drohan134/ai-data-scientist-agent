import json
from pathlib import Path

import pandas as pd

from tools.ml_analyzer import analyze_ml


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "ml_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_dataframe(case):

    if case["problem_type"] == "classification":

        data = {
            "age": [
                20, 25, 30, 35, 40,
                45, 50, 55, 60, 65
            ],

            case["target"]: [
                "No", "Yes", "No", "Yes", "No",
                "Yes", "No", "Yes", "No", "Yes"
            ]
        }

    else:

        data = {
            "area": [
                500, 600, 700, 800, 900, 1000,
                1100, 1200, 1300, 1400, 1500, 1600
            ],

            case["target"]: [
                10.0, 20.0, 31.0, 42.0, 55.0, 63.0,
                74.0, 86.0, 95.0, 108.0, 119.0, 130.0
            ]
        }

    return pd.DataFrame(data)


def evaluate_ml(case, actual):

    score = 0
    total = 4

    if actual["problem_type"] == case["problem_type"]:
        score += 1

    if actual["target_column"] == case["target"]:
        score += 1

    actual_models = list(actual["models"].keys())

    if set(actual_models) == set(case["expected_models"]):
        score += 1

    if case["problem_type"] == "classification":
        required_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        }
    else:
        required_metrics = {
            "mae",
            "r2_score"
        }

    metrics_valid = all(
        required_metrics.issubset(
            set(actual["models"][model].keys())
        )
        for model in actual_models
    )

    if metrics_valid:
        score += 1

    return (score / total) * 100


def run_ml_evaluation():

    test_cases = load_test_cases()
    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        df = create_dataframe(case)

        try:
            actual = analyze_ml(
                df,
                target_column=case["target"]
            )

            score = evaluate_ml(case, actual)

            print(f"  Expected: {case['problem_type']}")
            print(f"  Actual  : {actual['problem_type']}")
            print(f"  Score   : {score:.2f}%")

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
    print("ML ANALYSIS EVALUATION")
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
    run_ml_evaluation()