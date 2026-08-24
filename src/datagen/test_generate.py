"""Tiny-mesh smoke test: run on CPU with `JAX_PLATFORMS=cpu uv run python -m datagen.test_generate`."""

from pathlib import Path

import jax.random as jr
import neuralmag as nm
import numpy as np

from datagen.generate import (
    SimulationParams,
    init_m_random,
    init_m_s_state,
    init_m_uniform,
    main,
    random_H_ext,
    random_init,
    run,
)


def params(H_ext):
    return SimulationParams(
        n=(7, 7),
        dx=(5e-9, 5e-9, 3e-9),
        dt=10e-12,
        t_tot=5e-11,
        material={"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02},
        H_ext=H_ext,
        init="test",
    )


def test_run():
    key = jr.PRNGKey(0)
    inits = [
        lambda s: init_m_random(s, key),
        lambda s: init_m_s_state(s, inv=False),
        lambda s: init_m_uniform(s, [1, 0, 0]),
        random_init(key)[1],
    ]
    for init in inits:
        ms, coords = run(params(random_H_ext(key)), init)
        # 5 steps of 10 ps + the initial frame; nodal grid is (n + 1) per axis
        assert ms.shape == (6, 8, 8, 3), ms.shape
        assert [c.shape for c in coords] == [(8, 8), (8, 8)]
        norms = (ms**2).sum(-1) ** 0.5
        assert abs(norms - 1).max() < 1e-2, abs(norms - 1).max()  # integrator tol


def _first_m(init):
    """Initial magnetization only -- no fields registered, so this is cheap."""
    state = nm.State(nm.Mesh((7, 7), (5e-9, 5e-9, 3e-9)))
    init(state)
    return np.asarray(state.m.tensor)


def test_random_is_seeded():
    """Same key -> identical draws; different keys -> different ones."""
    k0, k1 = jr.split(jr.PRNGKey(0))
    assert random_H_ext(k0) == random_H_ext(k0)
    assert random_H_ext(k0) != random_H_ext(k1)

    assert np.array_equal(_first_m(random_init(k0)[1]), _first_m(random_init(k0)[1]))
    # over a fixed set of keys the choice actually varies (all draws unit-norm)
    assert random_init(k0)[0] == random_init(k0)[0]
    ms = [_first_m(random_init(k)[1]) for k in jr.split(jr.PRNGKey(0), 16)]
    assert len({m.tobytes() for m in ms}) > 1
    for m in ms:
        assert abs((m**2).sum(-1) ** 0.5 - 1).max() < 1e-5


def test_json_roundtrip():
    """dump -> from_json -> dump is a fixed point (JSON turns tuples into lists,
    so the reloaded object is compared through the file, not by ==)."""
    import tempfile

    p = params(random_H_ext(jr.PRNGKey(0)))
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        a.mkdir(), b.mkdir()
        p.json_dump(a)
        loaded = SimulationParams.from_json(a)
        loaded.json_dump(b)
        assert (a / "metadata.json").read_text() == (b / "metadata.json").read_text()
        assert loaded.init == p.init and list(loaded.n) == list(p.n)


def test_main_writes_samples():
    """End-to-end: main -> m.npy + metadata.json that reloads."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        main(seed=0, n_samples=2, base_dir=Path(d), n=(7, 7, 1), dx=(5e-9, 5e-9, 3e-9), t_tot=5e-11)
        for sample in ("sample_1", "sample_2"):
            out = Path(d) / sample
            assert np.load(out / "m.npy").shape == (6, 8, 8, 2, 3)
            assert SimulationParams.from_json(out).init


if __name__ == "__main__":
    test_run()
    test_random_is_seeded()
    test_json_roundtrip()
    test_main_writes_samples()
    print("OK")
