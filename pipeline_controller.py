import os
import sys
import time
import threading
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NODE_ORDER = [
    "profiler", "quality", "cleaning", "eda", "visualization",
    "preprocessing", "ml", "critic", "reporter"
]

PROG_MAP = {
    "profiler":      ( 5, 14, "🔍 Profiling dataset..."),
    "quality":       (14, 25, "🛡️ Analyzing data quality..."),
    "cleaning":      (25, 36, "🧹 Cleaning dataset..."),
    "eda":           (36, 50, "📈 Running exploratory analysis..."),
    "visualization": (50, 62, "📊 Generating visualizations..."),
    "preprocessing": (62, 73, "⚙️ Preprocessing features..."),
    "ml":            (73, 87, "🤖 Training & evaluating models..."),
    "critic":        (87, 94, "🔎 Validating results..."),
    "reporter":      (94,100, "📝 Generating final report..."),
}

_PIPELINE_LOCK = threading.Lock()

_STATE = {
    "thread": None,
    "is_running": False,
    "running_step": "",
    "progress": 0,
    "label": "Ready",
    "logs": [],
    "result": None,
    "completed_steps": [],
    "error": None,
}


def get_pipeline_state():
    """Return a thread-safe copy of the pipeline state dictionary."""
    with _PIPELINE_LOCK:
        return {
            "is_running": _STATE["is_running"],
            "running_step": _STATE["running_step"],
            "progress": _STATE["progress"],
            "label": _STATE["label"],
            "logs": list(_STATE["logs"]),
            "result": dict(_STATE["result"]) if _STATE["result"] is not None else None,
            "completed_steps": list(_STATE["completed_steps"]),
            "error": _STATE["error"],
        }


def is_pipeline_running():
    with _PIPELINE_LOCK:
        th = _STATE.get("thread")
        if th is not None and not th.is_alive() and _STATE["is_running"]:
            _STATE["is_running"] = False
            _STATE["running_step"] = ""
            if _STATE["progress"] < 100:
                _STATE["progress"] = 100
                _STATE["label"] = "✅ Analysis complete!"
        return _STATE["is_running"]


def reset_pipeline():
    with _PIPELINE_LOCK:
        _STATE["is_running"] = False
        _STATE["running_step"] = ""
        _STATE["progress"] = 0
        _STATE["label"] = "Ready"
        _STATE["logs"] = []
        _STATE["result"] = None
        _STATE["completed_steps"] = []
        _STATE["error"] = None
        _STATE["thread"] = None


def _pipeline_worker_fn(dataset_path_arg, preview_df_arg, target_column_arg=None):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print("=" * 60, flush=True)
    print("PIPELINE WORKER THREAD STARTED", flush=True)
    print(f"Dataset path: {dataset_path_arg}", flush=True)
    if target_column_arg:
        print(f"User Selected Target: {target_column_arg}", flush=True)
    print("=" * 60, flush=True)

    if preview_df_arg is None and dataset_path_arg and os.path.exists(dataset_path_arg):
        from tools.data_profiler import load_dataset
        try:
            preview_df_arg = load_dataset(dataset_path_arg)
        except Exception as e:
            print(f"[PIPELINE] Error reloading dataset: {e}", flush=True)

    logs = [{"m": "Pipeline initialized", "k": "info", "ts": time.strftime("%H:%M:%S")}]
    with _PIPELINE_LOCK:
        _STATE["logs"] = list(logs)
        _STATE["is_running"] = True
        _STATE["running_step"] = "profiler"
        _STATE["progress"] = 8
        _STATE["label"] = "Starting stage 1: Profiler..."

    try:
        from workflows.graph import build_graph
        from gemini_guard import reset_gemini_circuit_breaker
        reset_gemini_circuit_breaker()

        compiled_graph = build_graph()
        logs.append({"m": "LangGraph compiled successfully", "k": "done", "ts": time.strftime("%H:%M:%S")})
        with _PIPELINE_LOCK:
            _STATE["logs"] = list(logs)

        init_state = {
            "dataset_path":    dataset_path_arg,
            "dataframe":       preview_df_arg,
            "target_column":   target_column_arg,
            "completed_steps": [],
            "messages":        [],
        }
        accumulated_state = dict(init_state)

        for chunk in compiled_graph.stream(init_state):
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]
            accumulated_state.update(node_data)

            if "completed_steps" in node_data:
                accumulated_state["completed_steps"] = node_data["completed_steps"]
            steps = list(accumulated_state.get("completed_steps", []))
            if node_name not in steps:
                steps.append(node_name)
            accumulated_state["completed_steps"] = steps

            p0, p1, plbl = PROG_MAP.get(node_name, (50, 60, f"{node_name}..."))
            logs.append({"m": f"✓  {node_name.upper()} completed",
                         "k": "done", "ts": time.strftime("%H:%M:%S")})

            curr_idx  = NODE_ORDER.index(node_name) if node_name in NODE_ORDER else -1
            next_node = (NODE_ORDER[curr_idx + 1]
                         if 0 <= curr_idx < len(NODE_ORDER) - 1 else "")

            with _PIPELINE_LOCK:
                _STATE["result"] = dict(accumulated_state)
                _STATE["completed_steps"] = list(accumulated_state.get("completed_steps", []))
                if next_node:
                    _, _, next_lbl = PROG_MAP.get(next_node, (p1, p1, f"Running {next_node}..."))
                    logs.append({"m": f"▶  Starting {next_node.upper()}...",
                                 "k": "step", "ts": time.strftime("%H:%M:%S")})
                    _STATE["progress"] = p1
                    _STATE["label"] = next_lbl
                    _STATE["running_step"] = next_node
                else:
                    _STATE["progress"] = 100
                    _STATE["label"] = "Finalizing report..."
                    _STATE["running_step"] = ""
                _STATE["logs"] = list(logs)

        # Final completion write
        with _PIPELINE_LOCK:
            _STATE["result"] = dict(accumulated_state)
            _STATE["completed_steps"] = list(accumulated_state.get("completed_steps", []))
            _STATE["progress"] = 100
            _STATE["label"] = "✅ Analysis complete!"
            _STATE["running_step"] = ""
            _STATE["is_running"] = False
            _STATE["logs"] = list(logs)
        print("PIPELINE WORKER THREAD FINISHED SUCCESSFULLY!", flush=True)

    except Exception as exc:
        traceback.print_exc()
        logs.append({"m": f"✗ Error: {exc}", "k": "error", "ts": time.strftime("%H:%M:%S")})
        with _PIPELINE_LOCK:
            _STATE["running_step"] = ""
            _STATE["is_running"] = False
            _STATE["error"] = str(exc)
            _STATE["logs"] = list(logs)


def start_pipeline(dataset_path_arg, preview_df_arg, target_column_arg=None):
    """Launch the pipeline in a background thread if not already running."""
    with _PIPELINE_LOCK:
        th = _STATE.get("thread")
        if th is not None and th.is_alive():
            print("[PIPELINE] Pipeline already active, ignoring start request", flush=True)
            return False

        _STATE["is_running"] = True
        _STATE["running_step"] = "profiler"
        _STATE["progress"] = 5
        _STATE["label"] = "Compiling pipeline graph..."
        _STATE["logs"] = [{"m": "Pipeline starting...", "k": "step", "ts": time.strftime("%H:%M:%S")}]
        _STATE["result"] = None
        _STATE["completed_steps"] = []
        _STATE["error"] = None

        t = threading.Thread(
            target=_pipeline_worker_fn,
            args=(dataset_path_arg, preview_df_arg, target_column_arg),
            daemon=True,
        )
        _STATE["thread"] = t
        t.start()
        print("[PIPELINE] Thread launched successfully!", flush=True)
        return True
