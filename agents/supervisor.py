import os
import time
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Ollama model – change this to switch to a different local model.
# Make sure the model is pulled first:  ollama pull qwen2.5-coder:7b
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# =========================================================
# STATE
# =========================================================

class DataScientistState(TypedDict, total=False):
    dataset_path: str
    profile: Dict[str, Any]
    quality_report: Dict[str, Any]
    cleaning_report: Dict[str, Any]
    eda_report: Dict[str, Any]
    visualization_report: Dict[str, Any]
    ml_report: Dict[str, Any]
    critic_report: Dict[str, Any]
    target_column: str
    problem_type: str
    supervisor_decision: str
    completed_steps: List[str]
    messages: List[str]
    gemini_available: bool
    groq_available: bool


# =========================================================
# LLM
# =========================================================

def get_ollama_llm():
    """Primary LLM: local Qwen2.5-Coder via Ollama (fully offline, no API key)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=512,  # supervisor only needs a short response
    )


def get_llm():
    """Cloud fallback 1: Gemini API."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from gemini_guard import install_gemini_circuit_breaker

    install_gemini_circuit_breaker()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )


def get_groq_llm():
    """Cloud fallback 2: Groq API."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key
    )


def get_fallback_llm():
    return get_groq_llm()


def generate_manual_decision(completed_steps):
    return manual_fallback_decision(
        completed_steps,
        {},
        {}
    )

# =========================================================
# NORMALIZE COMPLETED STEPS
# =========================================================

def normalize_completed_steps(completed_steps: List[str]) -> set:
    normalized = set()

    for step in completed_steps:
        step = step.lower().strip()

        if "profil" in step:
            normalized.add("profiler")
        elif "quality" in step:
            normalized.add("quality")
        elif "clean" in step:
            normalized.add("cleaning")
        elif "eda" in step or "exploratory" in step:
            normalized.add("eda")
        elif "visual" in step:
            normalized.add("visualization")
        elif "preprocess" in step:
            normalized.add("preprocessing")
        elif (
            "machine learning" in step
            or step == "ml analysis"
            or step == "ml"
        ):
            normalized.add("ml")
        elif "critic" in step or "validation" in step:
            normalized.add("critic")
        elif "report" in step or "reporter" in step:
            normalized.add("reporter")

    return normalized


# =========================================================
# RULE-BASED FALLBACK
# =========================================================

def manual_fallback_decision(
    completed_steps: List[str],
    quality_report: Dict[str, Any],
    ml_report: Dict[str, Any]
) -> str:

    completed = normalize_completed_steps(completed_steps)

    pipeline = [
        ("profiler", "Data Profiling"),
        ("quality", "Data Quality Analysis"),
        ("cleaning", "Data Cleaning"),
        ("eda", "Exploratory Data Analysis"),
        ("visualization", "Visualization"),
        ("preprocessing", "Preprocessing"),
        ("ml", "Machine Learning Analysis"),
        ("critic", "Critic / Validation"),
        ("reporter", "Final Report"),
    ]

    for node, display_name in pipeline:
        if node not in completed:
            return (
                f"NEXT STEP: {display_name}\n\n"
                f"REASON: Rule-based fallback decision. "
                f"The selected stage has not yet been completed."
            )

    return (
        "NEXT STEP: Final Report\n\n"
        "REASON: All analysis and validation stages "
        "have been completed."
    )


# =========================================================
# LLM INVOKE WITH RETRY
# =========================================================

def _is_quota_or_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "429", "resource_exhausted", "quota", "rate limit", "rate_limit"
    ))


def invoke_with_retry(llm, prompt, attempts):
    """Invoke an LLM, but never retry a quota/rate-limit failure."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            print(f"  Attempt {attempt}/{attempts}")
            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                decision = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                )
            else:
                decision = response.content

            if decision:
                return decision.strip()

        except Exception as error:
            last_error = error
            print(f"  Failed: {error}")

            if _is_quota_or_rate_limit_error(error):
                print("  Quota/rate-limit detected. Switching provider.")
                raise error

            if attempt < attempts:
                time.sleep(1)

    if last_error:
        raise last_error

    return None


# =========================================================
# SUPERVISOR NODE
# =========================================================

def supervisor_node(state: Dict[str, Any]):

    print("\n" + "=" * 60)
    print("SUPERVISOR AGENT")
    print("=" * 60)

    completed_steps = state.get("completed_steps", [])

    print("\nCompleted steps:")

    if completed_steps:
        for step in completed_steps:
            print(f"  ✓ {step}")
    else:
        print("  None")

    profile = state.get("profile", {})
    quality_report = state.get("quality_report", {})
    cleaning_report = state.get("cleaning_report", {})
    eda_report = state.get("eda_report", {})
    visualization_report = state.get("visualization_report", {})
    ml_report = state.get("ml_report", {})
    critic_report = state.get("critic_report", {})

    target_column = state.get("target_column", "")
    problem_type = state.get("problem_type", "")

    normalized_completed = normalize_completed_steps(completed_steps)

    analysis_context = {
        "completed_steps": completed_steps,
        "profile": profile,
        "quality_report": quality_report,
        "cleaning_report": cleaning_report,
        "eda_report": eda_report,
        "visualization_report": visualization_report,
        "ml_report": ml_report,
        "critic_report": critic_report,
        "target_column": target_column,
        "problem_type": problem_type,
    }

    prompt = f"""
You are the Supervisor Agent in an AI Data Scientist system.

Your job is to decide WHICH SINGLE WORKFLOW STAGE should execute next.

You are controlling a LangGraph workflow.

COMPLETED STEPS:
{completed_steps}

Normalized completed stages:
{sorted(normalized_completed)}

AVAILABLE STAGES:
1. Data Profiling
2. Data Quality Analysis
3. Data Cleaning
4. Exploratory Data Analysis
5. Visualization
6. Preprocessing
7. Machine Learning Analysis
8. Critic / Validation
9. Final Report

RULES:
1. Do NOT invent data, statistics, metrics, or findings.
2. NEVER select a completed stage.
3. Follow the normal workflow order.
4. Preprocessing MUST be completed after Visualization
   and before Machine Learning Analysis.
5. The Preprocessing Agent decides which preprocessing
   operations are required.
6. Do NOT skip the Preprocessing stage.
7. If Machine Learning Analysis is not completed and
   preprocessing is completed, select it.
8. After core analysis, select Critic / Validation.
9. After Critic / Validation, select Final Report.
10. Return ONLY ONE next step.

ANALYSIS CONTEXT:
{analysis_context}

Return exactly:

NEXT STEP: <one stage>

REASON: <one short reason>
"""

    decision = None
    decision_source = None

    # Run-level circuit breakers. They reset automatically on the next graph run.
    ollama_available = state.get("ollama_available", True)
    gemini_available = state.get("gemini_available", True)
    groq_available = state.get("groq_available", True)

    # =====================================================
    # OLLAMA (PRIMARY – fully offline, no API key needed)
    # =====================================================

    if ollama_available:
        try:
            print(f"\nTrying Ollama ({OLLAMA_MODEL})...")
            t0 = time.time()
            decision = invoke_with_retry(
                get_ollama_llm(),
                prompt,
                attempts=2
            )
            elapsed = round(time.time() - t0, 2)

            if decision:
                decision_source = "ollama"
                print(f"\nOllama Supervisor Decision ({elapsed}s):")
                print("-" * 60)
                print(decision)

        except Exception as error:
            ollama_available = False
            print(f"\nOllama unavailable: {error}")
            print("Falling back to cloud LLMs...")
    else:
        print("\nOllama unavailable. Skipping.")

    # =====================================================
    # GEMINI (Cloud fallback 1)
    # =====================================================

    if not decision and gemini_available:
        try:
            print("\nTrying Gemini...")
            decision = invoke_with_retry(
                get_llm(),
                prompt,
                attempts=1
            )

            if decision:
                decision_source = "gemini"
                print("\nGemini Supervisor Decision:")
                print("-" * 60)
                print(decision)

        except Exception as error:
            gemini_available = False
            print("\nGemini unavailable for the rest of this run.")
            print(f"Reason: {error}")
    elif not decision:
        print("\nGemini circuit breaker active. Skipping Gemini.")

    # =====================================================
    # GROQ FALLBACK (Cloud fallback 2)
    # =====================================================

    if not decision and groq_available:
        try:
            print("\nTrying Groq fallback...")

            decision = invoke_with_retry(
                get_groq_llm(),
                prompt,
                attempts=1
            )

            if decision:
                decision_source = "groq"
                print("\nGroq Supervisor Decision:")
                print("-" * 60)
                print(decision)

        except Exception as error:
            groq_available = False
            print("\nGroq fallback unavailable for the rest of this run.")
            print(f"Reason: {error}")
    elif not decision:
        print("\nGroq circuit breaker active. Skipping Groq.")

    # =====================================================
    # MANUAL FALLBACK
    # =====================================================

    if not decision:
        print("\nGenerating rule-based manual decision...")

        decision = manual_fallback_decision(
            completed_steps,
            quality_report,
            ml_report
        )

        decision_source = "manual_fallback"

    # =====================================================
    # SAFETY CHECK
    # =====================================================

    decision_lower = decision.lower()
    requested_node = None

    if "data profiling" in decision_lower or "profiler" in decision_lower:
        requested_node = "profiler"

    elif "data quality" in decision_lower or "quality analysis" in decision_lower:
        requested_node = "quality"

    elif "data cleaning" in decision_lower or "cleaning" in decision_lower:
        requested_node = "cleaning"

    elif "exploratory data analysis" in decision_lower or "eda" in decision_lower:
        requested_node = "eda"

    elif "visualization" in decision_lower or "visualisation" in decision_lower:
        requested_node = "visualization"
    
    elif "preprocessing" in decision_lower or "preprocess" in decision_lower:
        requested_node = "preprocessing"

    elif "machine learning" in decision_lower or "ml analysis" in decision_lower:
        requested_node = "ml"

    elif "critic" in decision_lower or "validation" in decision_lower:
        requested_node = "critic"

    elif (
        "final report" in decision_lower
        or "reporter" in decision_lower
        or "report" in decision_lower
    ):
        requested_node = "reporter"

    if (
        requested_node is not None
        and requested_node in normalized_completed
    ):
        print(
            f"\nWARNING: Supervisor selected already "
            f"completed stage: {requested_node}"
        )

        print("Applying deterministic safety check...")

        decision = manual_fallback_decision(
            completed_steps,
            quality_report,
            ml_report
        )

        decision_source = f"{decision_source}_safety_override"

    # =====================================================
    # STORE DECISION
    # =====================================================

    print(
        "\nSupervisor Decision "
        f"(source: {decision_source}):"
    )

    print("-" * 60)
    print(decision)

    messages = state.get("messages", [])
    messages = messages + [decision]

    return {
        "supervisor_decision": decision,
        "messages": messages,
        "completed_steps": completed_steps + [
            f"supervisor decision ({decision_source})"
        ],
        "ollama_available": ollama_available,
        "gemini_available": gemini_available,
        "groq_available": groq_available,
    }