"""Run-level Gemini circuit breaker for the AI Data Scientist pipeline."""

import sys
from threading import Lock

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class GeminiCircuitOpenError(RuntimeError):
    """Raised when Gemini has already failed during the current run."""


_original_invoke = None
_original_ainvoke = None
_installed = False
_open = False
_lock = Lock()


def is_gemini_available():
    with _lock:
        return not _open


def reset_gemini_circuit_breaker():
    global _open
    with _lock:
        _open = False


def open_gemini_circuit_breaker(error=None):
    global _open
    with _lock:
        if not _open:
            _open = True
            if error is not None:
                print("\nGemini circuit breaker OPEN for this run.")
                print(f"Reason: {error}")


def _guarded_invoke(self, *args, **kwargs):
    if not is_gemini_available():
        raise GeminiCircuitOpenError(
            "Gemini circuit breaker is open. Gemini will not be called again in this run."
        )

    try:
        return _original_invoke(self, *args, **kwargs)
    except Exception as error:
        open_gemini_circuit_breaker(error)
        raise


async def _guarded_ainvoke(self, *args, **kwargs):
    if not is_gemini_available():
        raise GeminiCircuitOpenError(
            "Gemini circuit breaker is open. Gemini will not be called again in this run."
        )

    try:
        return await _original_ainvoke(self, *args, **kwargs)
    except Exception as error:
        open_gemini_circuit_breaker(error)
        raise


def install_gemini_circuit_breaker():
    global _installed, _original_invoke, _original_ainvoke

    if _installed:
        return True

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        # Local/deterministic pipeline stages can run without the optional
        # Gemini SDK. LLM access will still report its normal import error.
        return False

    _original_invoke = ChatGoogleGenerativeAI.invoke
    _original_ainvoke = ChatGoogleGenerativeAI.ainvoke

    ChatGoogleGenerativeAI.invoke = _guarded_invoke
    ChatGoogleGenerativeAI.ainvoke = _guarded_ainvoke
    _installed = True
    return True
