"""Run complete experiments and regenerate reports from recorded measurements."""
import argparse
import hashlib
import json
from pathlib import Path

from .runner import (append_record, configure_runtime, environment, read_records,
                     run_job, write_json)


class Session:
    def __init__(self, config, output, tasks):
        self.config, self.output, self.tasks = config, Path(output), tasks
        self.output.mkdir(parents=True, exist_ok=True)
        current_env = environment()
        source_root = Path(__file__).resolve().parents[1]
        source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(source_root.rglob("*.py")) if path.name != "report.py"}
        for name, value in (("config.json", config), ("tasks.json", tasks),
                            ("source_manifest.json", source_hashes)):
            path = self.output / name
            if path.exists() and json.loads(path.read_text()) != value:
                raise ValueError(f"{path} differs from this run; use a new output directory")
            write_json(path, value)
        env_path = self.output / "environment.json"
        if env_path.exists():
            if json.loads(env_path.read_text())["versions"] != current_env["versions"]:
                raise ValueError("Dependency versions changed; use a new output directory")
        else:
            write_json(env_path, current_env)
        self.tables = {}
        self.references = {}
        self.count = 0

    def table(self, filename):
        if filename not in self.tables:
            self.tables[filename] = {row["record_id"]: row for row in read_records(self.output / filename)}
        return self.tables[filename]

    def job(self, filename, payload, *, record_id, postprocess=None, **metadata):
        from .workloads import task_hash
        table = self.table(filename)
        if record_id in table:
            return table[record_id]
        options = dict(self.config.get("backend_options", {}))
        options["memory_bytes"] = self.config.get("memory_bytes", 2**31)
        payload = dict(payload, options=options)
        result = run_job(payload, self.config.get("timeout_s", 60), options["memory_bytes"])
        task = payload["task"]
        result.update(record_id=record_id, task_id=task["id"], family=task["family"],
                      task_hash=task_hash(task), **metadata)
        if postprocess:
            result = postprocess(result)
        append_record(self.output / filename, result)
        table[record_id] = result
        self.count += 1
        if self.count % 12 == 0 or result["status"] not in ("ok", "unsupported"):
            print(f"{self.config['name']}: {self.count} new jobs; {record_id}: {result['status']} ({result['total_s']:.3f}s)", flush=True)
        return result

    def prepare_references(self):
        for task in self.tasks:
            row = self.job("references.jsonl", {"operation": "reference", "task": task}, record_id=task["id"])
            self.references[task["id"]] = row if row["status"] == "ok" else None

    def amplitude(self, task, backend, repeat, strategy="native", filename="backends.jsonl"):
        from .validation import verify
        return self.job(filename, {"operation": "amplitude", "task": task,
                                  "backend": backend, "strategy": strategy},
                        record_id=f"{task['id']}/{backend}/{strategy}/{repeat}",
                        backend=backend, strategy=strategy, repeat=repeat,
                        postprocess=lambda row: verify(row, self.references.get(task["id"])))


def preprocess(session):
    for task in session.tasks:
        for repeat in range(session.config.get("repeats", 3)):
            # Rotate order deterministically to distribute gradual machine drift.
            strategies = ["native", "fixed", "adaptive"]
            strategies = strategies[repeat % 3:] + strategies[:repeat % 3]
            for strategy in strategies:
                session.amplitude(task, "quimb", repeat, strategy, "preprocess.jsonl")


def selection(session):
    from .selection import evaluate_selectors
    from .validation import verify
    for task in session.tasks:
        session.job("features.jsonl", {"operation": "features", "task": task}, record_id=task["id"])
        for repeat in range(session.config.get("repeats", 3)):
            backends = ["quimb", "statevector", "mitm"]
            backends = backends[repeat % 3:] + backends[:repeat % 3]
            for backend in backends:
                key = f"{task['id']}/{backend}/native/{repeat}"
                previous = session.table("preprocess.jsonl").get(key) if backend == "quimb" else None
                if previous and key not in session.table("backends.jsonl"):
                    copied = dict(previous, source="preprocess.jsonl; same task/backend/settings/repeat")
                    append_record(session.output / "backends.jsonl", copied)
                    session.table("backends.jsonl")[key] = copied
                session.amplitude(task, backend, repeat)
    assessment = evaluate_selectors(session.tasks,
                                    list(session.table("backends.jsonl").values()),
                                    list(session.table("features.jsonl").values()),
                                    timeout_s=session.config.get("timeout_s", 60))
    models = assessment.pop("models")
    write_json(session.output / "selection_models.json", models)
    write_json(session.output / "selection_oof.json", assessment)
    for task in session.tasks:
        for repeat in range(session.config.get("repeats", 3)):
            policies = ["fixed", "raw", "effective"]
            policies = policies[repeat % 3:] + policies[:repeat % 3]
            for policy in policies:
                session.job("selection_replay.jsonl",
                            {"operation": "select", "task": task, "policy": policy, "models": models},
                            record_id=f"{task['id']}/{policy}/{repeat}", policy=policy, repeat=repeat,
                            postprocess=lambda row, task=task: verify(row, session.references.get(task["id"])))


def batch(session):
    from .batch import generate_sign_vectors
    from .validation import verify
    tasks = []
    count_per_family = session.config.get("batch_tasks_per_family", 1)
    for family in ("iqp", "graphstate"):
        # At least one task at every scale, selected before any performance data.
        eligible = [task for task in session.tasks if task["family"] == family]
        scales = sorted({task["scale"] for task in eligible})
        for scale in scales:
            tasks.extend([task for task in eligible if task["scale"] == scale][:count_per_family])
    sizes = session.config.get("batch_sizes", [1, 8, 32, 128])
    for task in tasks:
        signs = generate_sign_vectors(task, max(sizes), seed=task.get("seed", 0) + 701)
        refs = session.job("batch_references.jsonl",
                           {"operation": "batch_reference", "task": task, "sign_vectors": signs},
                           record_id=task["id"])
        write_json(session.output / "batch_inputs" / f"{task['id']}.json",
                   {"task_id": task["id"], "sign_vectors": signs, "batch_sizes": sizes})

        def verify_batch(row):
            if row["status"] != "ok":
                row["verification"] = "not_completed"
                return row
            expected = refs.get("references", [])
            checks = [verify({"status": "ok", "amplitude": value}, expected[i] if i < len(expected) else None)
                      for i, value in enumerate(row["amplitudes"])]
            row["verification"] = "passed" if all(c["verification"] == "passed" for c in checks) else "reference_unavailable"
            row["verified_queries"] = sum(c["verification"] == "passed" for c in checks)
            row["max_absolute_error"] = max((c.get("absolute_error", 0) for c in checks), default=0)
            if any(c["status"] == "incorrect" for c in checks):
                row.update(status="incorrect", verification="failed")
            return row

        for size in sizes:
            for repeat in range(session.config.get("repeats", 3)):
                methods = ["independent", "shared_graph", "quimb_cached"]
                methods = methods[repeat % 3:] + methods[:repeat % 3]
                for method in methods:
                    session.job("batch.jsonl", {"operation": "batch", "task": task,
                                               "method": method, "sign_vectors": signs[:size]},
                                record_id=f"{task['id']}/{method}/{size}/{repeat}",
                                method=method, batch_size=size, repeat=repeat, postprocess=verify_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dataset", "preprocess", "select", "batch", "all", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    parser.add_argument("--tasks", type=Path, help="Use an existing JSON task array instead of the configured generators")
    args = parser.parse_args()
    configure_runtime()
    if args.command == "report":
        from .report import generate_report
        generate_report(args.output)
        return
    from .workloads import generate_tasks
    config = json.loads(args.config.read_text())
    tasks = json.loads(args.tasks.read_text()) if args.tasks else generate_tasks(config)
    session = Session(config, args.output, tasks)
    if args.command == "dataset":
        print(f"Saved {len(tasks)} tasks to {args.output / 'tasks.json'}")
        return
    session.prepare_references()
    if args.command in ("preprocess", "all"):
        preprocess(session)
    if args.command in ("select", "all"):
        selection(session)
    if args.command in ("batch", "all"):
        batch(session)
    from .report import generate_report
    generate_report(args.output)
    print(f"Finished {args.command}: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
