

# Install BEFORE importing agents that construct ChatGoogleGenerativeAI.
from typing import Any, Dict, List, TypedDict

from gemini_guard import install_gemini_circuit_breaker, reset_gemini_circuit_breaker

install_gemini_circuit_breaker()

from langgraph.graph import StateGraph, START, END

from workflows.nodes import (
    profiler_node,
    quality_node,
    cleaning_node,
    eda_node,
    visualization_node,
    preprocessing_node,
    ml_node,
)


def critic_node(state):
    from agents.critic import critic_node as run_critic
    return run_critic(state)


def reporter_node(state):
    from agents.reporter import reporter_node as run_reporter
    return run_reporter(state)


class DataScientistState(TypedDict, total=False):
    dataset_path: str
    dataframe: Any
    report_path: str
    html_report_path: str
    pdf_report_path: str

    profile: Dict[str, Any]
    quality_report: Dict[str, Any]
    cleaning_report: Dict[str, Any]
    eda_report: Dict[str, Any]
    visualization_report: Dict[str, Any]
    preprocessing_plan: Dict[str, Any]
    preprocessed_dataframe: Any
    preprocessed_dataset_path: str
    ml_report: Dict[str, Any]
    critic_report: Dict[str, Any]
    final_report: str

    target_column: str
    problem_type: str
    supervisor_decision: str
    supervisor_source: str

    completed_steps: List[str]
    messages: List[str]


def _normalize_step(text: str):
    text = text.lower().strip()
    if "profil" in text:
        return "profiler"
    if "quality" in text:
        return "quality"
    if "clean" in text:
        return "cleaning"
    if "eda" in text or "exploratory" in text:
        return "eda"
    if "visual" in text:
        return "visualization"
    if "preprocess" in text:
        return "preprocessing"
    if "machine learning" in text or "ml analysis" in text or text == "ml":
        return "ml"
    if "critic" in text or "validation" in text:
        return "critic"
    if "report" in text:
        return "reporter"
    return None


def _valid_initial_next_step(decision: str, completed_steps: List[str]):
    """Choose the first incomplete canonical stage, validating optional input."""
    completed = {_normalize_step(step) for step in completed_steps}
    completed.discard(None)

    requested = _normalize_step(decision or "")

    # Find the first incomplete stage in the canonical workflow.
    pipeline = [
        "profiler",
        "quality",
        "cleaning",
        "eda",
        "visualization",
        "preprocessing",
        "ml",
        "critic",
        "reporter",
    ]

    first_valid = next(
        (step for step in pipeline if step not in completed),
        "reporter",
    )

    if not decision or not decision.strip():
        return first_valid

    if requested not in pipeline or requested in completed:
        print(
            f"\nInvalid supervisor next step: {requested}. "
            f"Using valid next step: {first_valid}."
        )
        return first_valid

    # Do not permit the planner to skip required stages.
    requested_index = pipeline.index(requested)
    first_index = pipeline.index(first_valid)
    if requested_index > first_index:
        print(
            f"\nSupervisor attempted to skip required stage: {first_valid}. "
            f"Using valid next step: {first_valid}."
        )
        return first_valid

    return requested


def initial_route(state: DataScientistState):
    return _valid_initial_next_step(
        state.get("supervisor_decision", ""),
        state.get("completed_steps", []),
    )


def build_graph():
    graph_builder = StateGraph(DataScientistState)

    graph_builder.add_node("profiler", profiler_node)
    graph_builder.add_node("quality", quality_node)
    graph_builder.add_node("cleaning", cleaning_node)
    graph_builder.add_node("eda", eda_node)
    graph_builder.add_node("visualization", visualization_node)
    graph_builder.add_node("preprocessing", preprocessing_node)
    graph_builder.add_node("ml", ml_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("reporter", reporter_node)

    # The canonical workflow always starts at the first incomplete stage.
    # Calling an LLM supervisor here added network latency without changing
    # the route because initial_route rejects skipped stages.
    graph_builder.add_conditional_edges(
        START,
        initial_route,
        {
            "profiler": "profiler",
            "quality": "quality",
            "cleaning": "cleaning",
            "eda": "eda",
            "visualization": "visualization",
            "preprocessing": "preprocessing",
            "ml": "ml",
            "critic": "critic",
            "reporter": "reporter",
        },
    )

    # Deterministic canonical workflow after the one-time planner.
    graph_builder.add_edge("profiler", "quality")
    graph_builder.add_edge("quality", "cleaning")
    graph_builder.add_edge("cleaning", "eda")
    graph_builder.add_edge("eda", "visualization")
    graph_builder.add_edge("visualization", "preprocessing")
    graph_builder.add_edge("preprocessing", "ml")
    graph_builder.add_edge("ml", "critic")
    graph_builder.add_edge("critic", "reporter")
    graph_builder.add_edge("reporter", END)

    return graph_builder.compile()


if __name__ == "__main__":
    reset_gemini_circuit_breaker()
    graph = build_graph()

    initial_state: DataScientistState = {
        "dataset_path": "data/test_dataset.csv",
        "completed_steps": [],
        "messages": [],
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print("\nCompleted Steps:")
    for step in result.get("completed_steps", []):
        print(f"  ✓ {step}")

    print("\nCritic Report:")
    print(result.get("critic_report", "No critic report generated."))

    print("\nFinal Report:")
    print(result.get("final_report", "No final report generated."))
