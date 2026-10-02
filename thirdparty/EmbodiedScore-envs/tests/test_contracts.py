"""Gymnasium's env checker plus the contracts every benchmark shares (the
registry checks live in ``test_registry.py``). Needs habitat-sim, the datasets
and a GPU: skipped unless the data roots are set; a line whose split files are
not on disk is skipped."""

import os

import numpy as np
import pytest

pytest.importorskip("habitat_sim")
if not (os.environ.get("EMBODIEDSCORE_DATA_ROOT") and os.environ.get("EMBODIEDSCORE_SCENE_ROOT")):
    pytest.skip("EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT not set", allow_module_level=True)

from gymnasium.utils.env_checker import check_env

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks.env import Act

# one cheap split per line (both variants share it)
SPLITS = {"vlnce-r2r": "val_unseen", "vlnce-rxr": "val_unseen", "ivlnce": "val_unseen", "goat": "val_unseen"}


def split_of(name: str) -> str:
    return SPLITS[es.benchmark(name).line]


HABITAT = sorted(n for n, b in es.BENCHMARKS.items() if b.engine == "habitat")


@pytest.fixture(scope="module", params=HABITAT)
def stack(request):
    name = request.param
    try:
        env = es.make(name, split_of(name))
    except FileNotFoundError as e:
        pytest.skip(f"{name}: data not on disk ({e.filename})")
    yield name, env
    env.close()


def test_check_env_and_spaces(stack):
    name, env = stack
    b = es.benchmark(name)
    obs, info = env.reset(options={"episode": 0})
    assert env.observation_space.contains(obs), name
    assert obs["rgb"].dtype == np.uint8 and obs["rgb"].shape[:2] == (b.body.rgb.height, b.body.rgb.width)
    if b.body.depth is not None:
        assert obs["depth"].dtype == np.float32 and obs["depth"].shape[-1] == 1
    for k in ("position", "rotation", "heading", "pitch", "collided", "distance_to_goal", "stop_called", "goal_index", "episode", "goal"):
        assert k in info, (name, k)
    assert info["episode"]["index"] == 0 and info["stop_called"] is False
    if b.metrics is not None:
        assert "metrics" in info
    assert env.action_space.n == len(b.actions)
    check_env(env.unwrapped, skip_render_check=True)


def test_step_facts_and_termination(stack):
    name, env = stack
    b = es.benchmark(name)
    env.reset(options={"episode": 1})
    for a in (Act.FORWARD, Act.LEFT, Act.RIGHT):
        _, r, term, trunc, info = env.step(a)
        assert r == 0.0 and term is False and trunc is False
    if Act.LOOK_UP in b.actions:
        p0 = info["pitch"]
        _, _, _, _, info = env.step(Act.LOOK_UP)
        assert info["pitch"] > p0 - 1e-6
    _, _, term, trunc, info = env.step(Act.STOP)
    assert term is True and info["stop_called"] is True
    with pytest.raises(RuntimeError):
        env.step(Act.FORWARD)


def test_episode_selection_and_seeding(stack):
    name, env = stack
    _, info = env.reset(options={"episode": 2})
    assert info["episode"]["index"] == 2
    if es.benchmark(name).line != "ivlnce":   # tours address episodes explicitly
        a = env.reset(seed=123)[1]["episode"]["index"]
        b = env.reset(seed=123)[1]["episode"]["index"]
        assert a == b


@pytest.mark.parametrize("variant", ["standard", "upstream"])
def test_goal_sequence_advances(variant):
    env = es.make("goat", SPLITS["goat"], variant=variant)
    try:
        _, info = env.reset(options={"episode": 0})
        n = info["metrics"]["n_subtasks"]
        assert info["goal_index"] == 0 and n >= 5
        _, _, term, _, info = env.step(Act.SUBTASK_STOP)
        assert info["goal_index"] == 1 and info["subtask_closed"] is True and term is False
        assert len(info["metrics"]["subtask_success"]) == n
    finally:
        env.close()


def test_tour_transit():
    env = es.make("ivlnce", SPLITS["ivlnce"])
    try:
        _, info = env.reset(options={"episode": 0})
        assert info["tour"]["tour_index"] == 0 and info["tour"]["tour_protocol_ok"] is True
        _, _, term, _, info = env.step(Act.STOP)
        assert term and info["tour"]["tour_episodes_done"] == 1 and info["tour"]["tour_t_ndtw"] is not None
        _, info = env.reset(options={"episode": 1})
        assert info["tour"]["tour_index"] == 1 and info["tour"]["transit_steps"] > 0
        assert info["tour"]["tour_protocol_ok"] is True
        _, info = env.reset(options={"episode": 5})     # a jump breaks the protocol
        assert info["tour"]["tour_protocol_ok"] is False
    finally:
        env.close()
