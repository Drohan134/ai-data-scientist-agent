import json
from pathlib import Path

from agents.critic import critic_node


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "critic_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_state(case):
    return {
        **case["input"],
        "completed_steps": []
    }


def evaluate_critic(case, actual):
    expected = case["expected"]
    critic_report = actual.get("critic_report", {})

    findings = critic_report.get("findings", [])
    warnings = critic_report.get("warnings", [])
    recommendation = critic_report.get("recommendation", "")

    score_checks = []

    if "warning_contains" in expected:
        for text in expected["warning_contains"]:
            score_checks.append(
                any(text in warning for warning in warnings)
            )

    if "finding_contains" in expected:
        for text in expected["finding_contains"]:
            score_checks.append(
                any(text in finding for finding in findings)
            )

    if "warning_count" in expected:
        score_checks.append(
            len(warnings) == expected["warning_count"]
        )

    if "recommendation" in expected:
        score_checks.append(
            recommendation == expected["recommendation"]
        )

    if not score_checks:
        return 0.0

    return (
        sum(score_checks) / len(score_checks)
    ) * 100


def run_critic_evaluation():
    test_cases = load_test_cases()
    results = []

    for case in test_cases:
        print(f"\nRunning {case['case_id']}...")

        state = create_state(case)

        try:
            actual = critic_node(state)

            score = evaluate_critic(
                case,
                actual
            )

            print(f"  Score: {score:.2f}%")

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
    print("CRITIC / VALIDATION EVALUATION")
    print("=" * 60)

    print(f"Test Cases   : {len(results)}")
    print(f"Passed Cases : {passed}")
    print(f"Accuracy     : {overall_score:.2f}%")

    print("\nCase Results:")

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
    run_critic_evaluation()