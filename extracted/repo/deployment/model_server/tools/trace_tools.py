import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any

_TRACE_COUNTERS: dict[str, int] = {}
_TRUE_VALUES = {"1", "true", "yes", "on", "debug", "trace"}


def trace_enabled() -> bool:
    return os.getenv("STARVLA_TRACE", "").strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def should_trace(key: str, step: int | None = None, every: int | None = None) -> bool:
    if not trace_enabled():
        return False
    first_n = max(0, _env_int("STARVLA_TRACE_FIRST_N", 3))
    every = max(1, every or _env_int("STARVLA_TRACE_EVERY", 50))
    if step is not None:
        return step < first_n or step % every == 0
    count = _TRACE_COUNTERS.get(key, 0)
    _TRACE_COUNTERS[key] = count + 1
    return count < first_n or count % every == 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _sample_sequence(seq: Any, max_items: int, depth: int) -> list[Any]:
    return [summarize_value(v, max_items=max_items, depth=depth + 1) for v in list(seq)[:max_items]]


def summarize_value(value: Any, max_items: int | None = None, depth: int = 0) -> Any:
    max_depth = _env_int("STARVLA_TRACE_MAX_DEPTH", 3)
    max_items = max_items or _env_int("STARVLA_TRACE_MAX_ITEMS", 4)
    max_text = _env_int("STARVLA_TRACE_MAX_TEXT", 240)
    max_stats_numel = _env_int("STARVLA_TRACE_MAX_STATS_NUMEL", 50000)

    if depth > max_depth:
        return f"<{type(value).__name__}>"

    if isinstance(value, (str, int, float, bool)) or value is None:
        return _short_text(value, max_text) if isinstance(value, str) else value

    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            return {"type": "PIL.Image", "mode": value.mode, "size": list(value.size)}
    except Exception:
        pass

    try:
        import torch
        if isinstance(value, torch.Tensor):
            info: dict[str, Any] = {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": int(value.numel()),
                "requires_grad": bool(value.requires_grad),
            }
            if value.numel() > 0:
                detached = value.detach()
                flat = detached.reshape(-1)
                sample = flat[:max_items].float().cpu().tolist()
                info["sample"] = [_json_safe(v) for v in sample]
                if value.numel() <= max_stats_numel and not value.is_complex():
                    stats = detached.float()
                    info.update({
                        "min": _json_safe(stats.min().item()),
                        "max": _json_safe(stats.max().item()),
                        "mean": _json_safe(stats.mean().item()),
                    })
            return info
    except Exception as exc:
        return {"type": type(value).__name__, "summary_error": repr(exc)}

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            info = {
                "type": "np.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.size),
            }
            if value.size > 0:
                flat = value.reshape(-1)
                info["sample"] = [_json_safe(v) for v in flat[:max_items].tolist()]
                if value.size <= max_stats_numel and np.issubdtype(value.dtype, np.number):
                    arr = value.astype("float32", copy=False)
                    info.update({
                        "min": _json_safe(arr.min()),
                        "max": _json_safe(arr.max()),
                        "mean": _json_safe(arr.mean()),
                    })
            return info
        if isinstance(value, np.generic):
            return value.item()
    except Exception as exc:
        return {"type": type(value).__name__, "summary_error": repr(exc)}

    if isinstance(value, dict):
        keys = list(value.keys())
        shown = keys[:max_items]
        return {
            "type": "dict",
            "len": len(value),
            "keys": [str(k) for k in keys[: max_items * 2]],
            "items": {str(k): summarize_value(value[k], max_items=max_items, depth=depth + 1) for k in shown},
        }

    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": _sample_sequence(value, max_items=max_items, depth=depth),
        }

    if hasattr(value, "shape") or hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": _json_safe(getattr(value, "shape", None)),
            "dtype": _json_safe(getattr(value, "dtype", None)),
        }

    return _short_text(repr(value), max_text)


def trace(label: str, **fields: Any) -> None:
    if not trace_enabled():
        return
    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "label": label,
    }
    for key, value in fields.items():
        payload[key] = summarize_value(value)
    print("[STARVLA_TRACE] " + json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


@contextmanager
def trace_span(label: str, **fields: Any):
    start = time.time()
    trace(label + ".start", **fields)
    try:
        yield
    finally:
        trace(label + ".end", elapsed_s=time.time() - start, **fields)
