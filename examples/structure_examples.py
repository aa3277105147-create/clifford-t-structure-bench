"""Small exact examples that explain the structure metrics used by the benchmark."""
from __future__ import annotations

import json

from pfsparse.basis import xor_change
from pfsparse.graph_contraction import contract_graph, subdivide_edges_even
from pfsparse.obstructions import bounded_degree_graph_circuit, effective_graph_form
from pfsparse.pathsum import build_pathsum
from pfsparse.reduce import reduce_pathsum


def _form_summary(form):
    ps = form.ps
    return {
        "r": form.r,
        "t": form.t,
        "parity_supports": [list(p.support()) for p in ps.parities],
        "support_sum": form.support_sum,
        "linear_terms": [[j, coefficient] for j, coefficient in sorted(ps.a.items())],
        "quadratic_edges": [list(edge) for edge in sorted(ps.b)],
        "quadratic_edges_count": len(ps.b),
        "L": form.L,
        "exact_amplitude": ps.evaluate_exact(),
    }


def _degree(n, edges):
    degrees = [0] * n
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return max(degrees, default=0)


def variable_change_example():
    # <000| H^3 CZ_01 CZ_02 T^3 H^3 |000> has q=x0*x1+x0*x2.
    circuit = ([('H', (q,)) for q in range(3)]
               + [('CZ', (0, 1)), ('CZ', (0, 2))]
               + [('T', (q,)) for q in range(3)]
               + [('H', (q,)) for q in range(3)])
    original = reduce_pathsum(build_pathsum(circuit, 3, 0, 0))
    changed = xor_change(original, 1, 2)
    before, after = _form_summary(original), _form_summary(changed)
    return {
        "principle": "A variable change can move structural cost between the quadratic part and parity supports.",
        "quadratic_form": "q(x) = x0*x1 + x0*x2",
        "coordinate_change": "x0=u, x1=v XOR w, x2=w",
        "before": before,
        "after": after,
        "exact_amplitude_preserved": before["exact_amplitude"] == after["exact_amplitude"],
    }


def sparse_physical_dense_core_example():
    # The logical interaction graph is K4; the construction implements it with
    # a depth-five physical graph of maximum degree three.
    t = 4
    edges = [(u, v) for u in range(t) for v in range(u + 1, t)]
    physical = bounded_degree_graph_circuit(t, edges)
    compact = effective_graph_form(t, edges, root_output=0)
    independently_reduced = reduce_pathsum(
        build_pathsum(physical.circuit, physical.n, 0, physical.root_output(0)))
    exact_expected = compact.ps.evaluate_exact()
    exact_from_circuit = independently_reduced.ps.evaluate_exact()
    return {
        "principle": "Low physical interaction degree can coexist with a dense effective core.",
        "logical_graph": {"vertices": t, "edges": len(edges),
                          "density": len(edges) / (t * (t - 1) / 2)},
        "physical_circuit": {"qubits": physical.n, "gates": physical.s,
                             "depth": len(physical.layers),
                             "max_interaction_degree": _degree(physical.n, physical.physical_edges)},
        "effective_graph": {"vertices": compact.r, "edges": len(compact.ps.b),
                            "density": len(compact.ps.b) / (compact.r * (compact.r - 1) / 2),
                            "L": compact.L},
        "amplitude_check": {"method": "exact path-sum reduction versus the independently constructed graph form",
                            "physical_qubits": physical.n,
                            "live_variables_enumerated": compact.r,
                            "equal": exact_from_circuit == exact_expected,
                            "exact_amplitude": exact_expected},
    }


def boundary_contraction_example():
    # Subdivide every edge of a four-root cycle by 2, 4, 6, and 8 vertices.
    roots = 4
    root_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    rows = []
    exact_first = None
    for internal_pairs in range(1, 5):
        specification = subdivide_edges_even(roots, root_edges, internal_pairs)
        contracted = contract_graph(specification.n, specification.edges, specification.roots)
        row = {
            "internal_vertices_per_root_edge": 2 * internal_pairs,
            "physical_qubits": specification.n,
            "physical_gates": specification.gate_count,
            "physical_depth": len(specification.layers()),
            "physical_max_interaction_degree": _degree(specification.n, specification.edges),
            "root_boundary_vertices": contracted.t,
            "max_clifford_component_boundary": contracted.max_boundary,
            "contracted_variables": contracted.pf.r,
            "effective_L": contracted.pf.L,
        }
        if internal_pairs == 1:
            # At 12 physical qubits, compare contraction with a separate exact
            # path-sum build/reduction. Both final integer sums have four variables.
            direct = reduce_pathsum(build_pathsum(specification.circuit(), specification.n, 0, 0))
            expected = direct.ps.evaluate_exact(max_variables=16)
            actual = contracted.pf.ps.evaluate_exact(max_variables=16)
            exact_first = {"physical_qubits": specification.n,
                           "equal": actual == expected,
                           "exact_amplitude": actual}
        rows.append(row)
    return {
        "principle": "The Clifford interior can grow while its two-root-edge boundary and contracted representation stay small.",
        "root_graph": {"vertices": roots, "edges": len(root_edges)},
        "cases": rows,
        "exact_amplitude_check_for_smallest_case": exact_first,
    }


def main():
    result = {
        "title": "Three structural examples for Clifford+T amplitude simulation",
        "examples": [variable_change_example(),
                     sparse_physical_dense_core_example(),
                     boundary_contraction_example()],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
