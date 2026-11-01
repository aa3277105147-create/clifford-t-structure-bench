"""Preloaded dependencies; each query runs in its own measured child process."""
import json
import resource
import sys
import time
import traceback

import numpy  # preload the same numerical environment for every backend
import quimb.tensor
import qiskit

# Internal simplification contractions must obey the serial benchmark policy.
# Explicit, seeded cotengra planners still determine the final contraction path.
quimb.tensor.set_contract_strategy("greedy")

from .backends import evaluate
from .batch import evaluate_batch
from .selection import extract_features, select_backend
from .validation import reference


def execute(payload):
    operation = payload["operation"]
    task = payload["task"]
    options = payload.get("options", {})
    if operation == "amplitude":
        return evaluate(task, payload["backend"], payload.get("strategy", "native"), options)
    if operation == "reference":
        return reference(task)
    if operation == "batch_reference":
        records = []
        memo = {}
        for signs in payload["sign_vectors"]:
            key = tuple(signs)
            if key not in memo:
                iterator = iter(signs)
                variant = dict(task)
                variant["circuit"] = [[("T" if next(iterator) == 1 else "TDG") if gate in ("T", "TDG") else gate, qs]
                                      for gate, qs in task["circuit"]]
                memo[key] = reference(variant)
            records.append(memo[key])
        return {"references": records}
    if operation == "features":
        return extract_features(task)
    if operation == "batch":
        return evaluate_batch(task, payload["method"], payload["sign_vectors"], options)
    if operation == "select":
        route = select_backend(task, payload["models"], payload["policy"])
        answer = evaluate(task, route["selected_backend"], "native", options)
        answer["routing"] = route
        answer["backend"] = route["selected_backend"]
        return answer
    raise ValueError(f"unknown operation: {operation}")


def worker_main(connection, payload_json):
    payload = json.loads(payload_json)
    connection.send({"event": "ready"})
    started = time.perf_counter()
    try:
        payload["task"] = json.loads(payload.pop("task_json"))
        result = execute(payload)
        result.setdefault("status", "ok")
    except NotImplementedError as exc:
        result = {"status": "unsupported", "error": str(exc)}
    except MemoryError as exc:
        result = {"status": "memory_limit", "error": str(exc)}
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    result["total_s"] = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    connection.send({"event": "result", "result": result})
    connection.close()
