from __future__ import annotations

import cmath

import numpy as np
import pytest

from pfsparse.affine import Affine
from pfsparse.bench.backends import (
    _graph_form_tn,
    _parity_form_tn,
    build_graph_form,
    evaluate,
    quimb_contract,
    quimb_rehearse,
    build_quimb_circuit,
)
from pfsparse.parityform import ParityForm
from pfsparse.pathsum import PathSum
from pfsparse.reference import amplitude as reference_amplitude


def _task(circuit, n=3, x=1, y=5, global_phase=0.0):
    return {
        "id": "backend-check",
        "family": "test",
        "n": n,
        "circuit": [[gate, list(qubits)] for gate, qubits in circuit],
        "x": x,
        "y": y,
        "global_phase": global_phase,
        "metadata": {},
    }


def _complex(result):
    return complex(*result["amplitude"])


def _assert_close(actual, expected):
    assert abs(actual - expected) <= 1e-10 + 1e-8 * abs(expected)


def test_parity_form_tensor_export_keeps_long_parities_as_rank_three_xor_chains():
    ps = PathSum(
        nvars=5,
        alive={0, 1, 2, 3, 4},
        c=3,
        half_log2=-1,
        a={0: 1, 3: 2},
        b={(0, 4), (1, 3)},
        parities=[Affine(0b11111, 1), Affine(0b10110, 0)],
    )
    form = ParityForm.from_pathsum(ps)
    tn = _parity_form_tn(form)

    assert max(len(tensor.inds) for tensor in tn) <= 3
    value = tn.contract(all, output_inds=())
    _assert_close(complex(value), form.value())


@pytest.mark.parametrize("backend", ["quimb", "statevector", "mitm"])
def test_backends_match_exact_path_sum_with_nontrivial_phase_and_bit_order(backend):
    task = _task([
        ("H", (2,)),
        ("CNOT", (2, 0)),
        ("T", (0,)),
        ("S", (1,)),
        ("CZ", (0, 1)),
        ("TDG", (2,)),
        ("X", (1,)),
    ], global_phase=0.37)
    expected = reference_amplitude(
        [(gate, tuple(qs)) for gate, qs in task["circuit"]],
        task["n"], task["x"], task["y"],
    ) * cmath.exp(1j * task["global_phase"])
    result = evaluate(task, backend, options={"planning_repeats": 2})

    assert result["status"] == "ok"
    _assert_close(_complex(result), expected)
    assert result["timings_s"]["evaluate"] >= 0


def test_quimb_native_rehearsal_helpers_preserve_qiskit_bit_order():
    task = _task([("H", (0,)), ("CNOT", (0, 2)), ("T", (2,))], x=4, y=1)
    expected = reference_amplitude(
        [(gate, tuple(qs)) for gate, qs in task["circuit"]], 3, 4, 1)
    circuit = build_quimb_circuit(task)
    rehearsal = quimb_rehearse(circuit, task["y"], n=task["n"],
                               options={"planning_repeats": 2})
    actual = quimb_contract(rehearsal)["value"]

    _assert_close(actual, expected)


def test_graph_form_tensor_export_matches_mitm_for_slot_overrides():
    task = _task([
        ("H", (0,)), ("H", (1,)), ("CNOT", (0, 1)),
        ("T", (0,)), ("TDG", (1,)), ("CZ", (1, 2)),
    ], n=3, x=0, y=2)
    form = build_graph_form(task)
    signs = [-1, 1]
    tensor_network = _graph_form_tn(form, signs)
    tensornet_value = complex(tensor_network.contract(all, output_inds=()))
    mitm = form.mitm_value([np.exp(1j * np.pi / 4) ** sign for sign in signs])

    _assert_close(tensornet_value, mitm)


def test_adaptive_planner_is_serial_and_returns_a_valid_amplitude(monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from pfsparse.bench.workloads import generate_tasks

    def reject_process_pool(self, *args, **kwargs):
        raise AssertionError("single-threaded path planning must not create a process pool")

    monkeypatch.setattr(ProcessPoolExecutor, "__init__", reject_process_pool)
    task = generate_tasks({"families": ["grover"], "scales": [4], "seeds": [0]})[0]
    result = evaluate(
        task, "quimb", "adaptive",
        {"planning_repeats": 2, "local_rounds": 1, "local_samples": 4},
    )
    expected = reference_amplitude(
        [(gate, tuple(qs)) for gate, qs in task["circuit"]],
        task["n"], task["x"], task["y"],
    )

    assert result["status"] == "ok"
    assert {row["name"] for row in result["candidates"]} == {
        "native", "reduced", "rank", "local", "graph",
    }
    _assert_close(_complex(result), expected)
    assert result["timings_s"]["prepare"] + result["timings_s"]["plan"] > 0


def test_statevector_obeys_the_declared_memory_budget_without_allocating_the_state():
    task = _task([], n=24, x=0, y=0)
    result = evaluate(task, "statevector", options={"memory_bytes": 1024})

    assert result["status"] == "memory_limit"
    assert result["amplitude"] is None


@pytest.mark.parametrize("backend", ["quimb", "statevector", "mitm"])
def test_global_phase_is_applied_once_to_a_nonzero_amplitude(backend):
    task = _task([("H", (0,)), ("T", (0,))], n=1, x=0, y=0,
                 global_phase=0.37)
    expected = reference_amplitude(
        [(gate, tuple(qs)) for gate, qs in task["circuit"]], 1, 0, 0)
    expected *= cmath.exp(1j * task["global_phase"])
    assert abs(expected) > 1e-6

    result = evaluate(task, backend, options={"planning_repeats": 2})

    assert result["status"] == "ok"
    _assert_close(_complex(result), expected)

