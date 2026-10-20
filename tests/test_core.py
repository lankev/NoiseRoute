import numpy as np
import pytest

from noiseroute.core import PythonCore
from noiseroute.device import synthetic


def test_dependencies_and_permutation():
    c = PythonCore(3, [(0, 1), (1, 2)], [[0], [0, 2], [1], [2]], [0, 1, 2])
    assert c.drain() == [0, 2]
    assert c.front() == [1]
    c.swap(0)
    assert c.mapping() == [1, 0, 2]
    assert c.drain() == [1, 3]
    assert not c.remaining()


def test_rust_reference_parity():
    rust = pytest.importorskip("noiseroute_core")
    rng = np.random.default_rng(51)
    for kind in ["ring", "grid", "line", "neutral"]:
        d = synthetic(kind, 7)
        gates = [rng.choice(7, int(rng.integers(1, 3)), replace=False).tolist() for _ in range(70)]
        mapping = rng.permutation(7).tolist()
        p, r = [cls(7, d.edges, gates, mapping) for cls in (PythonCore, rust.RoutingCore)]
        for _ in range(300):
            assert p.drain() == r.drain()
            assert p.mapping() == r.mapping()
            assert p.front() == r.front()
            assert p.remaining() == r.remaining()
            assert p.mask() == r.mask()
            if not p.remaining():
                break
            action = int(rng.choice(np.flatnonzero(p.mask())))
            p.swap(action)
            r.swap(action)
        clone = r.clone()
        clone.swap(0)
        assert clone.mapping() != r.mapping()


@pytest.mark.parametrize("mapping", [[0, 0], [0], [0, 2]])
def test_invalid_permutation(mapping):
    with pytest.raises(ValueError):
        PythonCore(2, [(0, 1)], [[0, 1]], mapping)
