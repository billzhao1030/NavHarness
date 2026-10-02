"""The declarations and the gym registry: no simulator, no data needed
(split out of ``test_contracts.py``, whose module-level skip needs habitat-sim
and the data roots)."""

import pytest

import embodiedscore_envs as es


def test_registry_covers_every_declaration():
    import gymnasium as gym
    for b in es.BENCHMARKS.values():
        assert b.gym_id in gym.registry


def test_every_line_has_both_variants():
    from embodiedscore_envs.benchmarks import resolve
    from embodiedscore_envs.benchmarks.presets import actions, bodies, depth
    lines = {b.line for b in es.BENCHMARKS.values()}
    assert lines == {"vlnce-r2r", "vlnce-rxr", "ivlnce", "goat"}
    assert len(es.BENCHMARKS) == 2 * len(lines) == 8
    assert all(b.engine == "habitat" for b in es.BENCHMARKS.values())
    for line in lines:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert (std.variant, up.variant) == ("standard", "upstream") and std.line == up.line == line
        assert std.engine == up.engine
        assert std.body is bodies.STANDARD and std.actions in (actions.STANDARD, actions.GOAT)
        assert std.depth is depth.STANDARD
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")
        assert es.benchmark(f"{line}-upstream", "standard") is std
    assert resolve("goat-upstream") == "goat-upstream" and resolve("goat-upstream", "standard") == "goat"
    with pytest.raises(ValueError):
        resolve("goat", "legacy")
    with pytest.raises(KeyError):
        es.benchmark("goat-legacy")
