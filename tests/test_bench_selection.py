from pfsparse.bench.selection import evaluate_selectors, extract_features


def task(task_id, family, circuit):
    return {"id": task_id, "family": family, "n": 2, "circuit": circuit,
            "x": 0, "y": 0, "global_phase": 0.0}


def test_effective_features_are_for_the_requested_amplitude():
    item = task("a", "path", [["H", [0]], ["CNOT", [0, 1]], ["T", [1]]])
    features = extract_features(item)
    assert features["raw"] == {
        "n": 2, "gates": 3, "depth": 3, "t": 1,
        "interaction_edges": 1, "maxdegree": 1,
    }
    assert features["effective"]["r"] >= 0
    assert features["effective"]["L"] >= features["effective"]["r"]
    assert features["timings_s"]["effective_features"] >= 0


def test_leave_one_family_out_never_trains_on_heldout_family():
    tasks = [
        task("path-0", "path", [["H", [0]], ["CNOT", [0, 1]]]),
        task("grid-0", "grid", [["H", [0]], ["CZ", [0, 1]], ["T", [1]]]),
    ]
    feature_rows = [{"task_id": item["id"], **extract_features(item)} for item in tasks]
    records = []
    for item in tasks:
        for backend, elapsed in (("quimb", 1.0), ("statevector", 2.0), ("mitm", 3.0)):
            records.append({"task_id": item["id"], "family": item["family"],
                            "backend": backend, "status": "ok", "total_s": elapsed,
                            "verification": True})
    result = evaluate_selectors(tasks, records, feature_rows, timeout_s=60)
    assert result["models"]["folds"]["path"]["training_families"] == ["grid"]
    assert result["models"]["folds"]["grid"]["training_families"] == ["path"]
    assert len(result["predictions"]) == 6
    assert all(row["heldout_family"] == row["family"] for row in result["predictions"])
