"""Metrics for the displayed phase-factor graph, not a hardness certificate.

Min-fill returns an upper bound. A growing heuristic upper bound does not prove
that treewidth grows; only a bounded upper bound certifies that this graph is easy.
"""
from __future__ import annotations

from .pathsum import PathSum


def interaction_graph(ps: PathSum):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(ps.alive)
    g.add_edges_from(ps.b)
    for obj in list(ps.parities) + list(ps.constraints):
        support = obj.support()
        for i, u in enumerate(support):
            for v in support[i + 1:]:
                g.add_edge(u, v)
    return g


def treewidth_upper(ps: PathSum) -> int:
    """Min-fill upper bound for this representation (empty graph convention: 0)."""
    from networkx.algorithms.approximation import treewidth_min_fill_in
    g = interaction_graph(ps)
    if not g:
        return 0
    width, _ = treewidth_min_fill_in(g)
    return width


def bounded_graph_metrics(ps: PathSum, max_nodes: int = 128) -> dict:
    """Skip the heuristic above its size budget; max_nodes=0 disables it."""
    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")
    if max_nodes == 0 or len(ps.alive) > max_nodes:
        return {"graph_nodes": len(ps.alive), "graph_edges": None,
                "tw_upper": None, "tw_status": "skipped_size"}
    from networkx.algorithms.approximation import treewidth_min_fill_in
    g = interaction_graph(ps)
    if len(g) > max_nodes:
        return {"graph_nodes": len(g), "graph_edges": g.number_of_edges(),
                "tw_upper": None, "tw_status": "skipped_size"}
    width = treewidth_min_fill_in(g)[0] if g else 0
    return {"graph_nodes": len(g), "graph_edges": g.number_of_edges(),
            "tw_upper": width, "tw_status": "computed_upper_bound"}
