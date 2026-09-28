import json
from pathlib import Path
import re
from agents.supervisor import manual_fallback_decision


def load_test_cases():
    path = (
        Path(__file__).parent.parent
        / "datasets"
        / "workflow_cases.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_workflow_evaluation():

    test_cases = load_test_cases()
    results = []

    for case in test_cases:

        print(f"\nRunning {case['case_id']}...")

        actual_response = manual_fallback_decision(
        case["completed_steps"],
        {},
        {}
    )

        match = re.search(
        r"NEXT STEP:\s*(.+)",
        actual_response
    )

        actual_next = match.group(1).strip() if match else ""

        stage_mapping = {
        "Data Profiling": "Profiling",
        "Data Quality Analysis": "Quality",
        "Data Cleaning": "Cleaning",
        "Exploratory Data Analysis": "EDA",
        "Visualization": "Visualization",
        "Preprocessing": "Preprocessing",
        "Machine Learning Analysis": "ML",
        "Critic / Validation": "Critic",
        "Final Report": "Report"
    }

        actual_stage = stage_mapping.get(
        actual_next,
        actual_next
    )

        expected = case["expected_next"]

        score = 100.0 if actual_stage == expected else 0.0

        results.append({
        "case_id": case["case_id"],
        "expected": expected,
        "actual": actual_stage,
        "score": score
    })

        print(f"  Expected: {expected}")
        print(f"  Actual  : {actual_stage}")
        print(f"  Score   : {score:.2f}%")
    
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    passed = sum(1 for r in results if r["score"] == 100.0)
    
    return {
        "score": overall,
        "passed": passed,
        "total": len(results)
    }

if __name__ == "__main__":
    run_workflow_evaluation()