"""Compare graph forms with stored R1--R4 records on identical circuits/outputs.

No new output is selected. Source rows without a selected output stay censored.
Timing is descriptive and includes no claim of a controlled performance study.
"""
from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path

from .experiments import make_circuit, _deadline, ExperimentTimeout
from .pathsum import build_pathsum
from .reduce import reduce_pathsum, compress_to_rank
from .graph_form import graph_normal_form, circuit_graph_normal_form
from .biclique import compress_bicliques


FIELDS = ("family", "size", "seed", "n", "s", "t", "output_y_hex", "source_file",
          "source_status", "source_verification", "status", "stage", "rank_L", "ls_L",
          "residual_vertices", "residual_edges", "physical_vertices", "physical_edges",
          "physical_fourier", "physical_half_log2", "candidate_edges", "compressed_vertices",
          "compressed_edges", "compression_used", "bicliques", "incidence_weight", "block_size",
          "verification", "seconds", "note")


def run_source(row, source, timeout=45):
    out = {k: row[k] for k in ("family", "size", "seed", "n", "s", "t", "output_y_hex") if k in row}
    out.update(source_file=source.name, source_status=row["status"],
               source_verification=row["output_verification"], rank_L=row["rank_L"], ls_L=row["ls_L"],
               stage="source", status="running", verification="not_run")
    if not row.get("output_y_hex"):
        out.update(status="source_censored_no_output", seconds=0, note="original row had no selected output")
        return out
    started = time.perf_counter()
    try:
        with _deadline(timeout):
            family, size, seed = row["family"], int(row["size"]), int(row["seed"])
            circuit, n = make_circuit(family, size, random.Random(seed), float(row["width_exponent"]), int(row["brick_depth"]))
            y = int(row["output_y_hex"], 16)
            out["stage"] = "rank"
            pf = compress_to_rank(reduce_pathsum(build_pathsum(circuit, n, 0, y), order=row["order"]))
            if int(row["rank_L"]) != pf.L:
                raise AssertionError("stored and regenerated rank L differ")
            out["stage"] = "residual_graph"
            residual = graph_normal_form(pf)
            out.update(residual_vertices=residual.n, residual_edges=len(residual.edges))
            out["stage"] = "physical_graph"
            physical = circuit_graph_normal_form(circuit, n, 0, y, order=row["order"])
            out.update(physical_vertices=physical.n, physical_edges=len(physical.edges),
                       physical_fourier=sum(label.kind == "fourier" for label in physical.labels),
                       physical_half_log2=physical.half_log2)
            assert physical.n == int(row["t"])
            out["stage"] = "biclique"
            compression = compress_bicliques(physical)
            out.update(candidate_edges=compression.candidate_edges,
                       compressed_vertices=compression.form.n, compressed_edges=len(compression.form.edges),
                       compression_used=compression.used, bicliques=len(compression.partition),
                       incidence_weight=compression.incidence_weight, block_size=compression.block_size)
            out["stage"] = "verification"
            if physical.n <= 10:
                expected = pf.ps.evaluate_exact(max_variables=14)
                assert physical.evaluate_exact() == expected
                assert residual.evaluate_exact() == expected
                out["verification"] = "exact_cyclotomic"
            elif physical.n <= 20 and pf.r <= 14:
                d, coeff = pf.ps.evaluate_exact(max_variables=14)
                from .graph_form import OMEGA
                expected = 2.0 ** (-d) * sum(a * OMEGA ** j for j, a in enumerate(coeff))
                actual = physical.mitm_value()
                if abs(actual - expected) > 1e-9 * max(1, abs(expected)):
                    raise AssertionError("MITM differs from exact parity oracle")
                out["verification"] = "numerical_mitm_vs_exact_parity"
            else:
                out["verification"] = "algebraic_construction_only"
            out.update(stage="complete", status="ok")
    except ExperimentTimeout:
        out.update(status="timeout", note="graph comparison wall-time budget exhausted")
    out["seconds"] = round(time.perf_counter() - started, 6)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=["F0", "F1", "F2", "F3", "F4"], required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--timeout", type=float, default=45)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for suffix in ("pilot", "extension"):
            source = args.results / f"{args.family.lower()}_{suffix}.csv"
            with source.open() as original:
                for row in csv.DictReader(original):
                    out = run_source(row, source, args.timeout)
                    writer.writerow(out)
                    fh.flush()
                    print(f"{out['family']} {out['size']} seed={out['seed']} {out['status']} {out['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
