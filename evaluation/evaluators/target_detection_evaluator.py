import json
import pandas as pd

from agents.target_detection_agent import target_detection_agent
import agents.target_detection_agent as target_module


def load_test_cases():
    with open(
        "evaluation/datasets/target_detection_cases.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def create_dataframe(case):
    data = {}

    for feature in case["features"]:
        data[feature] = [1, 2, 3, 4]

    # Target is always the last column
    if case["problem_type"] == "classification":
        data[case["target"]] = ["No", "Yes", "No", "Yes"]
    else:
        data[case["target"]] = [10.5, 20.5, 30.5, 40.5]

    return pd.DataFrame(data)


def run_target_detection_evaluation():

    test_cases = load_test_cases()

    # Don't consume Gemini quota during evaluation
    target_module.GEMINI_EVAL_DISABLED = True

    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        df = create_dataframe(case)

        actual = target_detection_agent(df)

        target_correct = (
            actual["target"] == case["target"]
        )

        problem_correct = (
            actual["problem_type"] == case["problem_type"]
        )

        score = (
            (int(target_correct) + int(problem_correct)) / 2
        ) * 100

        results.append({
            "case_id": case["case_id"],
            "score": score
        })

        print(
            f"  Target: "
            f"{'✓' if target_correct else '✗'} "
            f"expected={case['target']} "
            f"actual={actual['target']}"
        )

        print(
            f"  Problem: "
            f"{'✓' if problem_correct else '✗'} "
            f"expected={case['problem_type']} "
            f"actual={actual['problem_type']}"
        )

        print(f"  Score: {score:.2f}%")

    overall = (
        sum(r["score"] for r in results)
        / len(results)
    )

    passed = sum(
        r["score"] == 100
        for r in results
    )

    print("\n" + "=" * 60)
    print("TARGET DETECTION EVALUATION")
    print("=" * 60)

    print(f"Test Cases   : {len(results)}")
    print(f"Passed Cases : {passed}")
    print(f"Accuracy     : {overall:.2f}%")

    for result in results:
        status = "PASS" if result["score"] == 100 else "PARTIAL"
        print(
            f"{result['case_id']} : "
            f"{result['score']:.2f}% [{status}]"
        )
    return {
        "score": overall,
        "passed": passed,
        "total": len(results)
    }
    

if __name__ == "__main__":
    run_target_detection_evaluation()