"""Resource-limited jobs and append-only, inspectable experiment records."""
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import psutil


def configure_runtime():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "QUIMB_NUM_THREAD_WORKERS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    cache = Path(".cache").resolve()
    for name, folder in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib")):
        (cache / folder).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache / folder)
    mp.set_forkserver_preload(["pfsparse.bench.worker"])


def environment():
    names = ("numpy", "quimb", "cotengra", "qiskit", "networkx", "psutil", "matplotlib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "versions": {name: importlib.metadata.version(name) for name in names},
            "threads": 1, "process_start_method": "forkserver",
            "timing_scope": "JSON task parsing through route/preprocess/plan/evaluate; common dependency import/startup separately recorded",
            "memory_scope": "individual job process peak RSS, including shared imported libraries; parent/forkserver excluded",
            "memory_enforcement": "RSS sampled every 5 ms plus final OS high-water mark; polling can overshoot the threshold",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_job(payload, timeout_s=60, memory_bytes=2**31):
    from .worker import worker_main
    context = mp.get_context("forkserver")
    receive, send = context.Pipe(duplex=False)
    payload = dict(payload)
    payload["task_json"] = json.dumps(payload.pop("task"))
    process = context.Process(target=worker_main, args=(send, json.dumps(payload)), daemon=False)
    launched = time.perf_counter()
    process.start()
    send.close()
    observer = psutil.Process(process.pid)
    ready_at = None
    peak = 0
    result = None
    while result is None:
        if receive.poll(0.005):
            try:
                event = receive.recv()
            except EOFError:
                break
            if event["event"] == "ready":
                ready_at = time.perf_counter()
            else:
                result = event["result"]
                break
        try:
            peak = max(peak, observer.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        now = time.perf_counter()
        if peak > memory_bytes:
            result = {"status": "memory_limit", "error": "job RSS exceeded budget"}
        elif now - (ready_at or launched) > timeout_s:
            result = {"status": "timeout", "error": "job exceeded time budget"}
        elif not process.is_alive() and not receive.poll():
            break
    if process.is_alive() and (result is None or result.get("status") in ("timeout", "memory_limit")):
        process.terminate()
    process.join()
    finished = time.perf_counter()
    receive.close()
    if result is None:
        result = {"status": "error", "error": f"worker exited without result (exit={process.exitcode})"}
    result.setdefault("total_s", finished - (ready_at or launched))
    result["peak_rss_bytes"] = max(peak, result.get("peak_rss_bytes", 0))
    result["startup_s"] = (ready_at or finished) - launched
    result["process_wall_s"] = finished - launched
    if result["status"] == "ok" and result["peak_rss_bytes"] > memory_bytes:
        result["status"] = "memory_limit"
    if result["status"] == "ok" and result["total_s"] > timeout_s:
        result["status"] = "timeout"
    result.update(timeout_s=timeout_s, memory_budget_bytes=memory_bytes)
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_record(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_records(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
