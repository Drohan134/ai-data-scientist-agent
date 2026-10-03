import time
import pandas as pd

from agents.supervisor import get_ollama_llm, get_llm, get_fallback_llm, generate_manual_decision
from tools.data_profiler import load_dataset

def profiler_node(state):
    print("\n" + "=" * 60)
    print("DATA PROFILER NODE")
    print("=" * 60)

    # Streamlit passes its already-parsed preview dataframe to avoid reading
    # and parsing the uploaded file a second time. CLI callers still load it.
    df = state.get("dataframe")
    if df is None:
        df = load_dataset(state["dataset_path"])

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": df.columns.tolist(),
        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
            if count > 0
        }
    }

    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")

    return {
        "profile": profile,
        "dataframe": df,
        "completed_steps": state.get("completed_steps", [])
        + ["data profiling"]
    }


def quality_node(state):
    from tools.data_quality import analyze_data_quality

    print("\n" + "=" * 60)
    print("DATA QUALITY NODE")
    print("=" * 60)

    df = state["dataframe"]

    report = analyze_data_quality(df)

    print(f"Quality Score: {report['quality_score']}/100")

    return {
        "quality_report": report,
        "completed_steps": state.get("completed_steps", [])
        + ["data quality analysis"]
    }


def cleaning_node(state):
    from tools.data_cleaner import clean_data

    print("\n" + "=" * 60)
    print("DATA CLEANING NODE")
    print("=" * 60)

    df = state["dataframe"]

    cleaned_df, cleaning_report = clean_data(df)

    cleaned_path = "data/cleaned_dataset.csv"

    cleaned_df.to_csv(
        cleaned_path,
        index=False
    )

    print(f"Cleaned dataset saved to: {cleaned_path}")

    return {
        "dataframe": cleaned_df,
        "cleaning_report": cleaning_report,
        "dataset_path": cleaned_path,
        "completed_steps": state.get("completed_steps", [])
        + ["data cleaning"]
    }


def eda_node(state):
    from tools.eda_analyzer import analyze_eda

    print("\n" + "=" * 60)
    print("EDA NODE")
    print("=" * 60)

    df = state["dataframe"]

    id_columns = state.get(
        "quality_report",
        {}
    ).get(
        "id_columns",
        []
    )

    report = analyze_eda(
        df,
        id_columns=id_columns
    )

    print(
        f"Dataset shape: "
        f"{report['dataset_shape']}"
    )

    print(
        f"Strong correlations found: "
        f"{len(report['strong_correlations'])}"
    )

    return {
        "eda_report": report,
        "completed_steps": state.get("completed_steps", [])
        + ["EDA"]
    }


def visualization_node(state):
    from tools.visualization import generate_visualizations

    print("\n" + "=" * 60)
    print("VISUALIZATION NODE")
    print("=" * 60)

    report = generate_visualizations(
        state["dataframe"]
    )

    return {
        "visualization_report": report,
        "completed_steps": state.get("completed_steps", []) + ["visualization"]
    }

def preprocessing_node(state):
    from agents.preprocessing_agent import preprocessing_agent
    from tools.preprocessing import apply_preprocessing

    print("\n" + "=" * 60)
    print("PREPROCESSING NODE")
    print("=" * 60)

    df = state.get("dataframe")

    if df is None:
        raise ValueError("Dataframe not available for preprocessing.")

    # Agent decides WHAT preprocessing is required
    plan = preprocessing_agent(
        df,
        profile=state.get("profile"),
        quality_report=state.get("quality_report")
    )

    # Executor applies the agent's decision
    preprocessed_df = apply_preprocessing(df, plan)

    # Save preprocessed dataset
    output_path = "data/preprocessed_dataset.csv"
    preprocessed_df.to_csv(output_path, index=False)

    print(f"\n[OK] Preprocessed dataset saved to: {output_path}")

    return {
        "preprocessing_plan": plan,
        "preprocessed_dataframe": preprocessed_df,
        "preprocessed_dataset_path": output_path,
        "completed_steps": state.get("completed_steps", [])
        + ["Preprocessing"]
    }


def ml_node(state):
    from tools.ml_analyzer import analyze_ml

    print("\n" + "=" * 60)
    print("ML ANALYSIS NODE")
    print("=" * 60)

    df = state["dataframe"]

    target_column = state.get("target_column")

    if not target_column:
        potential_targets = state.get(
            "quality_report",
            {}
        ).get(
            "potential_target_columns",
            []
        )

        # Prefer churn if available.
        if "churn" in potential_targets:
            target_column = "churn"
        elif potential_targets:
            target_column = potential_targets[0]

    report = analyze_ml(
        df,
        target_column=target_column
    )

    print(
        f"\nTarget: "
        f"{report['target_column']}"
    )

    print(
        f"Problem type: "
        f"{report['problem_type']}"
    )

    print(
        f"Best model: "
        f"{report['best_model']}"
    )

    return {
        "ml_report": report,
        "target_column": report["target_column"],
        "problem_type": report["problem_type"],
        "completed_steps": state.get("completed_steps", [])
        + ["ML analysis"]
    }


def supervisor_node(state):
    print("\n" + "=" * 60)
    print("SUPERVISOR AGENT")
    print("=" * 60)

    completed_steps = state.get("completed_steps", [])

    print("\nCompleted steps:")
    for step in completed_steps:
        print(f"  [OK] {step}")

    prompt = f"""
You are the Supervisor Agent of an AI Data Scientist system.

Completed steps:
{completed_steps}

Available next steps:
- data profiling
- data quality analysis
- data cleaning
- EDA
- visualization
- ML analysis
- Critic / Validation
- Final Report

Choose exactly ONE next step.

Rules:
1. Do not invent results.
2. Do not repeat a completed step unless validation requires it.
3. After ML analysis, choose Critic / Validation.
4. After Critic / Validation, choose Final Report.
5. Return your decision in this format:

NEXT STEP: <one step>

REASON: <short explanation>
"""

    decision = None
    source = "ollama"

    # ---------------------------------------------------------
    # 1. Try Ollama (primary – offline)
    # ---------------------------------------------------------

    try:
        from agents.supervisor import OLLAMA_MODEL
        print(f"\nTrying Ollama ({OLLAMA_MODEL})...")
        t0 = time.time()
        llm = get_ollama_llm()
        response = llm.invoke(prompt)

        if isinstance(response.content, list):
            decision = "".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
            )
        else:
            decision = response.content

        if not decision or not decision.strip():
            raise ValueError("Ollama returned an empty response.")

        elapsed = round(time.time() - t0, 2)
        print(f"Ollama succeeded in {elapsed}s.")
        source = "ollama"

    except Exception as ollama_error:
        print(f"\nOllama supervisor unavailable: {ollama_error}")
        print("Falling back to cloud LLMs...")

        # ---------------------------------------------------------
        # 2. Try Gemini (cloud fallback 1)
        # ---------------------------------------------------------

        try:

            llm = get_llm()

            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                decision = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                )
            else:
                decision = response.content

            if not decision or not decision.strip():
                raise ValueError("Gemini returned an empty response.")

            source = "gemini"
            print("Gemini fallback succeeded.")

        except Exception as gemini_error:

            print(
                f"\nWARNING: Gemini supervisor unavailable."
            )
            print(f"Reason: {gemini_error}")

            # -------------------------------------------------
            # 3. Try Groq (cloud fallback 2)
            # -------------------------------------------------

            print("\nTrying Groq fallback...")

        try:

            llm = get_fallback_llm()

            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                decision = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                )
            else:
                decision = response.content

            if not decision or not decision.strip():
                raise ValueError("Groq returned an empty response.")

            source = "groq"

            print("Groq fallback succeeded.")

        except Exception as groq_error:

            print(
                f"WARNING: Groq fallback also unavailable."
            )
            print(f"Reason: {groq_error}")

            # -----------------------------------------
            # 3. Manual / rule-based fallback (last resort)
            # -----------------------------------------

            print(
                "\nGenerating rule-based manual decision..."
            )

            decision = generate_manual_decision(state)

            source = "manual_fallback"

            print(
                "Manual fallback decision generated successfully."
            )

    print(
        f"\nSupervisor Decision (source: {source}):"
    )
    print("-" * 60)
    print(decision)

    return {
        "supervisor_decision": decision,
        "supervisor_source": source,
        "messages": state.get("messages", []) + [decision]
    }