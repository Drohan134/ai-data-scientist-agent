import json
from pathlib import Path

import pandas as pd

from agents.preprocessing_agent import preprocessing_agent
import agents.preprocessing_agent as preprocessing_module
import gemini_guard

preprocessing_module.GEMINI_EVAL_DISABLED = True

FIELDS = [
    "imputation",
    "scaling",
    "encoding",
    "transformation"
]


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "preprocessing_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_test_dataframe(case):
    """
    Creates a small DataFrame representing the characteristics
    described in the evaluation case.
    """

    data = {
        "age": [20, 25, 30, 35, 40],
        "income": [1000, 2000, 3000, 4000, 5000]
    }

    df = pd.DataFrame(data)

    case_input = case["input"]

    # Add categorical feature when required
    if case_input["categorical_features"]:
        df["city"] = [
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            "Mumbai"
        ]

    # Add missing value when required
    if case_input["missing_values"]:
        df.loc[2, "income"] = None

    # Create substantially different numeric scale
    if case_input["numeric_scale_difference"]:
        df["large_value"] = [
            10000,
            20000,
            30000,
            40000,
            50000
        ]

    # Create highly skewed feature
    if case_input["extreme_skew"]:
        df["skewed_feature"] = [
            1,
            1,
            1,
            1,
            100000
        ]

    return df


def evaluate_prediction(expected, actual):

    correct = 0
    details = {}

    for field in FIELDS:

        expected_value = expected.get(field)

        actual_field = actual.get(field, {})

        if isinstance(actual_field, dict):
            actual_value = actual_field.get("required")
        else:
            actual_value = actual_field

        is_correct = expected_value == actual_value

        if is_correct:
            correct += 1

        details[field] = {
            "expected": expected_value,
            "actual": actual_value,
            "correct": is_correct
        }

    total = len(FIELDS)

    score = (correct / total) * 100

    return {
        "correct": correct,
        "total": total,
        "score": score,
        "details": details
    }

def run_preprocessing_evaluation():

    test_cases = load_test_cases()

    results = []

    print("\n" + "=" * 60)
    print("RUNNING ACTUAL PREPROCESSING AGENT EVALUATION")
    print("=" * 60)

    for case in test_cases:

        case_id = case["case_id"]

        print(f"\nRunning {case_id}...")

        try:
            # --------------------------------------------------
            # 1. Create test dataset
            # --------------------------------------------------
            df = build_test_dataframe(case)

            # --------------------------------------------------
            # 2. Run actual preprocessing agent
            # --------------------------------------------------
            actual = preprocessing_agent(
                df=df,
                profile=None,
                quality_report=None
            )

            # --------------------------------------------------
            # 3. Compare agent output with expected output
            # --------------------------------------------------
            evaluation = evaluate_prediction(
                case["expected"],
                actual
            )
            print(f"Score    : {evaluation['score']:.2f}%")
            
            for field, detail in evaluation["details"].items():

                symbol = "✓" if detail["correct"] else "✗"

                print(
                f"  {symbol} {field}: "
                f"expected={detail['expected']} "
                f"actual={detail['actual']}"
                 )

            results.append({
                "case_id": case_id,
                "status": "PASS"
                if evaluation["score"] == 100
                else "PARTIAL",
                "score": evaluation["score"],
                "correct": evaluation["correct"],
                "total": evaluation["total"],
                "expected": case["expected"],
                "actual": actual,
                "details":evaluation["details"]
            })

            print(
                f"Expected : {case['expected']}"
            )

            print(
                f"Actual   : {actual}"
            )

            print(
                f"Score    : "
                f"{evaluation['score']:.2f}%"
            )
            for field, detail in evaluation["details"].items():

                symbol = "✓" if detail["correct"] else "✗"

            print(
                f"  {symbol} {field}: "
                f"expected={detail['expected']} "
                f"actual={detail['actual']}"
             )

        except Exception as error:

            print(
                f"ERROR in {case_id}: {error}"
            )

            results.append({
                "case_id": case_id,
                "status": "ERROR",
                "score": 0.0,
                "correct": 0,
                "total": len(FIELDS),
                "expected": case["expected"],
                "actual": None,
                "error": str(error)
            })

    # ------------------------------------------------------
    # Overall score
    # ------------------------------------------------------

    total_correct = sum(
        result["correct"]
        for result in results
    )

    total_possible = sum(
        result["total"]
        for result in results
    )

    overall_score = (
        total_correct / total_possible
    ) * 100 if total_possible else 0

    passed_cases = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    return {
        "evaluator": "preprocessing",
        "cases": len(test_cases),
        "passed_cases": passed_cases,
        "score": overall_score,
        "results": results
    }


if __name__ == "__main__":
    
    result = run_preprocessing_evaluation()

    print("\n" + "=" * 60)
    print("PREPROCESSING EVALUATION RESULT")
    print("=" * 60)

    print(
        f"Test Cases   : {result['cases']}"
    )

    print(
        f"Passed Cases : {result['passed_cases']}"
    )

    print(
        f"Accuracy     : {result['score']:.2f}%"
    )

    print("\nCase Results:")

    for case in result["results"]:

        print(
            f"{case['case_id']} : "
            f"{case['score']:.2f}% "
            f"[{case['status']}]"
        )