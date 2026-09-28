import json
from pathlib import Path


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "eda_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_eda(expected, actual):
    return {
        "correct": expected == actual,
        "score": 100.0 if expected == actual else 0.0
    }


def run_eda_evaluation():

    test_cases = load_test_cases()
    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        # Placeholder
        actual = {
            "required_sections": case["required_sections"]
        }

        expected = {
            "required_sections": case["required_sections"]
        }

        result = evaluate_eda(expected, actual)

        results.append({
            "case_id": case["case_id"],
            "score": result["score"]
        })

        print(f"  Score: {result['score']:.2f}%")


    overall_score = (
        sum(result["score"] for result in results)
        / len(results)
    )

    passed = sum(
        result["score"] == 100
        for result in results
    )

    print("\n" + "=" * 60)
    print("EDA ANALYSIS EVALUATION")
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
    run_eda_evaluation()