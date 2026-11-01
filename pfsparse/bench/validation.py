"""Independent reference records, computed outside benchmark timing."""
import cmath

from pfsparse.pathsum import build_pathsum
from pfsparse.reduce import reduce_pathsum
from pfsparse.reference import amplitude

ATOL = 1e-10
RTOL = 1e-8


def reference(task):
    circuit, n, x, y = task["circuit"], task["n"], task["x"], task["y"]
    phase = cmath.exp(1j * task.get("global_phase", 0))
    pf = reduce_pathsum(build_pathsum(circuit, n, x, y), order="min_degree")
    result = {"reference_kind": "unavailable", "amplitude": None}
    if pf.ps.zero or len(pf.ps.alive) <= 16:
        denominator, coefficients = pf.ps.evaluate_exact(max_variables=16)
        value = phase * 2.0**(-denominator) * sum(
            a * cmath.exp(1j * j * cmath.pi / 4) for j, a in enumerate(coefficients)
        )
        result.update(reference_kind="exact_reduced_pathsum", amplitude=[value.real, value.imag],
                      exact={"denominator_power": denominator, "coefficients": coefficients,
                             "global_phase_radians": task.get("global_phase", 0)})
    if n <= 16:
        independent = phase * amplitude(circuit, n, x, y)
        result["independent_statevector"] = [independent.real, independent.imag]
        if result["amplitude"] is not None:
            error = abs(independent - complex(*result["amplitude"]))
            if error > ATOL + RTOL * abs(independent):
                raise ArithmeticError(f"exact and independent references disagree: {error}")
            result["reference_crosscheck_error"] = error
        else:
            result.update(reference_kind="independent_numpy_statevector",
                          amplitude=[independent.real, independent.imag])
    return result


def verify(record, expected):
    if record["status"] != "ok":
        record["verification"] = "not_completed"
        return record
    if not expected or expected.get("amplitude") is None:
        record["verification"] = "reference_unavailable"
        return record
    value, target = complex(*record["amplitude"]), complex(*expected["amplitude"])
    error = abs(value - target)
    passed = error <= ATOL + RTOL * abs(target)
    record.update(verification="passed" if passed else "failed", absolute_error=error,
                  reference_kind=expected["reference_kind"])
    if not passed:
        record["status"] = "incorrect"
    return record
