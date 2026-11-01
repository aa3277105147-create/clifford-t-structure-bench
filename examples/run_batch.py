"""Run a user-specified T/T† batch from a JSON task description."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfsparse.bench.runner import configure_runtime, run_job
from pfsparse.bench.workloads import task_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON containing a task and T/T† sign vectors")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    task = config["task"]
    method = config["method"]
    options = dict(config.get("options", {}))
    options["memory_bytes"] = int(config["memory_bytes"])
    result = run_job(
        {"operation": "batch", "task": task, "method": method,
         "sign_vectors": config["sign_vectors"], "options": options},
        timeout_s=int(config["timeout_s"]), memory_bytes=options["memory_bytes"],
    )
    result.update(task_id=task["id"], task_hash=task_hash(task), method=method,
                  sign_vectors=config["sign_vectors"], verification="not_checked")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    configure_runtime()
    main()
