import itertools

import numpy as np
import pytest

from cyberfly.lif import Sim
from cyberfly.neurons import Circuit
from cyberfly.runtime import boot_kicks, period


def simulate(c: Circuit, cycles: int):
    return Sim(*c.arrays()).run(cycles * c.M * period(), boot_kicks(c))


def test_clock_ring_is_exact():
    c = Circuit(20)
    t, i = simulate(c, 500)
    p = period()
    for k in range(20):
        fired = t[i == c.ring[k]]
        assert np.array_equal(fired, 1 + k * p + np.arange(len(fired)) * 20 * p)


def test_latches_hold():
    c = Circuit(20)
    values = [0, 1, 1, 0]
    latches = [c.latch(v) for v in values]
    t, i = simulate(c, 200)
    for li, v in zip(latches, values):
        a, b = c.latches[li]
        assert (i == a).sum() == (2000 if v else 0)
        assert (i == b).sum() == (0 if v else 2000)


@pytest.mark.parametrize("fan_in", [1, 2, 3])
def test_nor_truth_table(fan_in):
    for lags in itertools.product([1, 2], repeat=fan_in):
        for k in (12, 13):
            for bits in itertools.product([0, 1], repeat=fan_in):
                c = Circuit(20)
                inputs = []
                for lag, v in zip(lags, bits):
                    ph = k - lag
                    a, b = c.latches[c.latch(v if ph % 2 == 0 else 1 - v)]
                    inputs.append(a if ph % 2 == 0 else b)
                g = c.nor(inputs, k, list(lags))
                t, i = simulate(c, 3)
                expected = [] if any(bits) else [1 + (k + 20 * cy) * period() for cy in range(3)]
                assert t[i == g].tolist() == expected
