"""pfsparse: parity-form size L for Clifford+T amplitudes (Path B framework).

Implemented: affine helpers, reference simulator, circuit families, PathSum/ParityForm
containers with brute-force evaluation, metrics, experiment harness.
To implement (see DESIGN.md): pathsum.build_pathsum, reduce.*, basis.*, slice_route.*.
"""
from .affine import Affine
from .pathsum import PathSum, build_pathsum
from .parityform import ParityForm

__all__ = ["Affine", "PathSum", "build_pathsum", "ParityForm"]
