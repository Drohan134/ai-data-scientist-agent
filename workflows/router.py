def route_after_supervisor(state):

    decision = state.get(
        "supervisor_decision",
        ""
    ).lower()

    completed = [
        step.lower()
        for step in state.get(
            "completed_steps",
            []
        )
    ]

    # -------------------------------------------------
    # Map supervisor decision to a node
    # -------------------------------------------------

    if "profil" in decision:
        target = "profiler"

    elif "quality" in decision:
        target = "quality"

    elif "clean" in decision:
        target = "cleaning"

    elif "eda" in decision or "exploratory data analysis" in decision:
        target = "eda"

    elif "visual" in decision:
        target = "visualization"

    elif "preprocess" in decision:
        target = "preprocessing"

    elif "machine learning" in decision or "ml analysis" in decision:
        target = "ml"

    elif "critic" in decision or "validation" in decision:
        target = "critic"

    elif "report" in decision:
        target = "reporter"

    else:
        target = "critic"

    # -------------------------------------------------
    # Completed stages
    # -------------------------------------------------

    completed_map = {
        "profiler": "data profiling",
        "quality": "data quality analysis",
        "cleaning": "data cleaning",
        "eda": "eda",
        "visualization": "visualization",
        "preprocessing": "preprocessing",
        "ml": "machine learning",
        "critic": "critic / validation",
        "reporter": "final report",
    }

    completed_names = set(completed)

    # -------------------------------------------------
    # Prevent already completed stages from repeating
    # -------------------------------------------------

    if target in completed_map:

        required_name = completed_map[target]

        if required_name in completed_names:

            pipeline = [
                ("profiler", "data profiling"),
                ("quality", "data quality analysis"),
                ("cleaning", "data cleaning"),
                ("eda", "eda"),
                ("visualization", "visualization"),
                ("preprocessing", "preprocessing"),
                ("ml", "machine learning"),
                ("critic", "critic / validation"),
                ("reporter", "final report"),
            ]

            for node, name in pipeline:
                if name not in completed_names:
                    return node

            return "reporter"

    # -------------------------------------------------
    # Safety: Reporter should only run after Critic
    # -------------------------------------------------

    if target == "reporter":
        if "critic / validation" not in completed_names:
            return "critic"

    return target