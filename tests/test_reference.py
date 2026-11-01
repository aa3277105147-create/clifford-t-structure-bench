import numpy as np

from pfsparse.reference import statevector, amplitude, W8
from pfsparse.pathsum import PathSum
from pfsparse.affine import Affine


def test_single_gates():
    assert np.allclose(statevector([("H", (0,))], 1, 0), [2 ** -0.5, 2 ** -0.5])
    assert np.allclose(statevector([("H", (0,))], 1, 1), [2 ** -0.5, -(2 ** -0.5)])
    assert np.isclose(amplitude([("T", (0,))], 1, 1, 1), W8)
    assert np.isclose(amplitude([("TDG", (0,))], 1, 1, 1), W8 ** 7)
    assert np.isclose(amplitude([("S", (0,))], 1, 1, 1), 1j)
    assert np.isclose(amplitude([("CNOT", (0, 1))], 2, 0b01, 0b11), 1)
    assert np.isclose(amplitude([("CZ", (0, 1))], 2, 0b11, 0b11), -1)


def test_hs_global_phase():
    circ = [("H", (0,)), ("S", (0,))] * 3          # (HS)^3 = omega * I
    assert np.isclose(amplitude(circ, 1, 0, 0), W8)


def test_pathsum_bruteforce_conventions():
    # sum_{z0} omega^{2*1*z0} with one parity [z0]:  1 + i * omega
    ps = PathSum()
    j = ps.new_var()
    ps.add_a(j, 1)
    ps.parities.append(Affine.var(j))
    assert np.isclose(ps.evaluate_bruteforce(), 1 + 1j * W8)
    # a constraint z0 = 1 keeps one term; half_log2 = -1 multiplies by 2^{-1/2}
    ps.constraints.append(Affine(1 << j, 1))
    ps.half_log2 = -1
    assert np.isclose(ps.evaluate_bruteforce(), (1j * W8) / np.sqrt(2))
