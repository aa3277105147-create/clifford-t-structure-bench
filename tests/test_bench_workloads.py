import cmath
import json
import math
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pfsparse.pathsum import build_pathsum
from pfsparse.reduce import reduce_pathsum
from pfsparse.bench.workloads import (
    NATIVE_GATES,
    generate_batch_queries,
    generate_tasks,
    load_qasm,
    task_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def _statevector_amplitude(task):
    qc = QuantumCircuit(task["n"])
    for gate, qubits in task["circuit"]:
        q = qubits[0] if qubits else None
        if gate == "H":
            qc.h(q)
        elif gate == "X":
            qc.x(q)
        elif gate == "Z":
            qc.z(q)
        elif gate == "S":
            qc.s(q)
        elif gate == "SDG":
            qc.sdg(q)
        elif gate == "T":
            qc.t(q)
        elif gate == "TDG":
            qc.tdg(q)
        elif gate == "CNOT":
            qc.cx(*qubits)
        elif gate == "CZ":
            qc.cz(*qubits)
        else:
            raise AssertionError(gate)
    qc.global_phase = task["global_phase"]
    state = Statevector.from_int(task["x"], dims=1 << task["n"]).evolve(qc)
    return state.data[task["y"]]


def test_formal_workload_is_stable_complete_and_unique():
    config = json.loads((ROOT / "configs/formal.json").read_text())
    tasks = generate_tasks(config)
    again = generate_tasks(config)

    assert len(tasks) == 240
    assert [task["sha256"] for task in tasks] == [task["sha256"] for task in again]
    assert len({task["sha256"] for task in tasks}) == 240
    assert all(task["sha256"] == task_hash(task) for task in tasks)
    assert all(gate in NATIVE_GATES for task in tasks for gate, _ in task["circuit"])
    assert {family: sum(task["family"] == family for task in tasks)
            for family in config["families"]} == {family: 40 for family in config["families"]}


def test_grover_and_adder_queries_match_their_known_amplitudes():
    grover = generate_tasks({"families": ["grover"], "scales": [4], "seeds": [3]})[0]
    grover_expected_magnitude = math.sin(3 * math.asin(1 / math.sqrt(1 << 4)))
    assert math.isclose(abs(_statevector_amplitude(grover)), grover_expected_magnitude,
                        rel_tol=1e-12, abs_tol=1e-12)

    adder = generate_tasks({"families": ["arithmetic"], "scales": [4], "seeds": [101]})[0]
    width = adder["metadata"]["adder_width"]
    chosen_a = adder["metadata"]["selected_superposition_branch"]
    phase_pattern = adder["metadata"]["input_phase_pattern"]
    phase_exponent = sum(((phase_pattern >> (2 * q)) & 3) * ((chosen_a >> q) & 1)
                         for q in range(width))
    expected = cmath.exp(1j * math.pi * phase_exponent / 2) / math.sqrt(1 << width)
    assert abs(_statevector_amplitude(adder) - expected) < 1e-12


def test_subdivided_graphstate_keeps_four_t_slots_and_matches_exact_pathsum():
    task = generate_tasks({"families": ["graphstate"], "scales": [6], "seeds": [4]})[0]
    assert task["n"] == 12
    assert task["metadata"]["root_count"] == 4
    assert task["metadata"]["internal_path_vertices_per_edge"] == 2
    assert sum(gate in ("T", "TDG") for gate, _ in task["circuit"]) == 4
    assert all(gate == "H" for gate, _ in task["circuit"][:task["n"]])
    assert all(gate == "H" for gate, _ in task["circuit"][-task["n"]:])

    circuit = [(gate, tuple(qubits)) for gate, qubits in task["circuit"]]
    reduced = reduce_pathsum(build_pathsum(circuit, task["n"], task["x"], task["y"]))
    assert len(reduced.ps.alive) == task["metadata"]["root_count"]
    d, coefficients = reduced.ps.evaluate_exact()
    omega = cmath.exp(1j * math.pi / 4)
    exact = sum(coefficients[i] * omega ** i for i in range(4)) / (1 << d)
    assert abs(_statevector_amplitude(task) - exact) < 1e-10


def test_batch_fillings_are_precomputed_and_repeatable():
    task = generate_tasks({"families": ["iqp"], "scales": [4], "seeds": [8]})[0]
    first = generate_batch_queries(task, 32, seed=9)
    assert first == generate_batch_queries(task, 32, seed=9)
    slot_count = sum(gate in ("T", "TDG") for gate, _ in task["circuit"])
    assert all(len(query["slot_signs"]) == slot_count for query in first)
    assert all(sign in (-1, 1) for query in first for sign in query["slot_signs"])


def test_qasm_import_removes_terminal_measurement_and_hashes_input(tmp_path):
    path = tmp_path / "tiny.qasm"
    source = (
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; '
        'h q[0]; t q[0]; measure q[0] -> c[0];'
    )
    path.write_text(source)
    task = load_qasm(path, x=0, y=1)

    assert task["n"] == 1
    assert task["circuit"] == [["H", [0]], ["T", [0]]]
    assert task["x"] == 0 and task["y"] == 1
    assert task["metadata"]["qasm_sha256"]
    assert task["sha256"] == task_hash(task)
    assert abs(_statevector_amplitude(task) - cmath.exp(1j * math.pi / 4) / math.sqrt(2)) < 1e-12


def test_qasm_import_rejects_approximate_rotation_synthesis(tmp_path):
    path = tmp_path / "approximate.qasm"
    path.write_text(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; rx(pi/7) q[0];'
    )
    try:
        load_qasm(path)
    except ValueError as exc:
        assert "outside the supported exact Clifford+T" in str(exc)
    else:
        raise AssertionError("arbitrary-angle Rx must not be synthesized approximately")
