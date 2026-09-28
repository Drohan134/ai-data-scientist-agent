from evaluation.evaluators.preprocessing_evaluator import (
    run_preprocessing_evaluation
)

from evaluation.evaluators.target_detection_evaluator import (
    run_target_detection_evaluation
)

from evaluation.evaluators.workflow_evaluator import (
    run_workflow_evaluation
)


def main():

    print("\n")
    print("=" * 70)
    print("AI DATA SCIENTIST - MLEVALS")
    print("=" * 70)

    preprocessing = run_preprocessing_evaluation()

    target_detection = run_target_detection_evaluation()

    workflow = run_workflow_evaluation()

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nPreprocessing Accuracy : "
        f"{preprocessing['score']:.2f}%"
    )

    print(
        f"Target Detection      : "
        f"{target_detection['score']:.2f}%"
    )

    print(
        f"Workflow Validity     : "
        f"{workflow['score']:.2f}%"
    )

    overall_score = (
        preprocessing["score"]
        + target_detection["score"]
        + workflow["score"]
    ) / 3

    print(
        f"\nOverall Evaluation    : "
        f"{overall_score:.2f}%"
    )

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()