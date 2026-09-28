import json
import pandas as pd

from agents.supervisor import get_llm, get_fallback_llm


GEMINI_EVAL_DISABLED = False


def target_detection_agent(df):

    print("\n" + "=" * 60)
    print("TARGET DETECTION AGENT")
    print("=" * 60)

    columns = df.columns.tolist()

    # MLEval mode
    if GEMINI_EVAL_DISABLED:

        print("\nMLEVAL MODE: Gemini disabled.")

        target = columns[-1]

        if pd.api.types.is_numeric_dtype(df[target]):
            problem_type = "regression"
        else:
            problem_type = "classification"

        return {
            "target": target,
            "problem_type": problem_type
        }

    prompt = f"""
You are the Target Detection Agent of an AI Data Scientist system.

Dataset columns:
{columns}

Determine:
1. The most likely target column.
2. Whether the problem is classification or regression.

Return ONLY valid JSON:

{{
    "target": "...",
    "problem_type": "classification"
}}
"""

    try:
        response = get_llm().invoke(prompt)
        decision = response.content

    except Exception as gemini_error:

        print("Gemini target detection unavailable.")
        print(f"Reason: {gemini_error}")

        response = get_fallback_llm().invoke(prompt)
        decision = response.content

    if isinstance(decision, list):
        decision = "".join(
            item.get("text", "")
            for item in decision
            if isinstance(item, dict)
        )

    decision = decision.strip()

    if decision.startswith("```"):
        decision = decision.replace("```json", "")
        decision = decision.replace("```", "").strip()

    return json.loads(decision)