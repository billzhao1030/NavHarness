"""The blue line is a working note, not the archive.

The map holds the geometry for the whole visit. The drawn trajectory does not:
by the sixth sub-goal a whole-visit line is a ball of blue string across every
room and says nothing about what the robot has just been doing. So the line is
scoped to the current sub-goal, inside the current episode — and everything
earlier stays in ``est_track``, recorded, simply not redrawn.

Two boundaries, because they are not the same boundary:

  * ``task_track0`` — a sub-goal closed (GOAT: 5-10 per episode, no reset);
  * ``ep_track0``  — a new episode. Today a fresh SlamEnv is built per package
    episode so this is 0, but a tour keeps one map and ``load_spatial``
    restores the visit's whole track with it; the slice is what stops that
    quietly putting the entire tour back on the picture.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sen = importlib.import_module("exp_workspace.lhnav_es.mcp._slam_env")


def walker():
    """A SlamEnv stub with only the track bookkeeping on it."""
    se = object.__new__(sen.SlamEnv)
    se.est_track = []
    se.task_track0 = 0
    se.ep_track0 = 0
    se.tick = 0
    return se


def walk(se, n: int) -> None:
    for _ in range(n):
        se.est_track.append([float(len(se.est_track)), 0.0, 0.0])
        se.tick += 1


def new_episode(se) -> None:
    """What reset() does to the track bookkeeping."""
    se.ep_track0 = len(se.est_track)
    se.task_track0 = se.ep_track0


def test_a_closed_subgoal_starts_the_line_over():
    se = walker()
    walk(se, 5)
    assert len(se.drawn_track()) == 5
    se.begin_task(1)
    assert se.drawn_track() == [], "a new sub-goal draws from here"
    walk(se, 3)
    assert [p[0] for p in se.drawn_track()] == [5.0, 6.0, 7.0]


def test_earlier_subgoals_are_recorded_not_redrawn():
    se = walker()
    walk(se, 4)
    se.begin_task(1)
    walk(se, 4)
    assert len(se.drawn_track()) == 4
    assert len(se.est_track) == 8, "the archive keeps everything"


def test_a_new_episode_ends_the_line_even_mid_subgoal():
    se = walker()
    walk(se, 6)                       # an episode's worth of walking
    new_episode(se)
    assert se.drawn_track() == []
    walk(se, 2)
    assert [p[0] for p in se.drawn_track()] == [6.0, 7.0]
    assert len(se.est_track) == 8


def test_a_carried_track_does_not_put_the_whole_tour_back_on_the_picture():
    """The case this slice exists for: the track is restored with the map."""
    se = walker()
    walk(se, 40)                      # a previous visit, loaded back in
    new_episode(se)                   # …and this episode begins
    walk(se, 3)
    assert len(se.drawn_track()) == 3
    assert len(se.est_track) == 43


def test_the_later_boundary_wins():
    se = walker()
    walk(se, 10)
    se.begin_task(1)                  # sub-goal boundary at 10
    walk(se, 5)
    se.ep_track0 = 12                 # an episode boundary AFTER it
    assert len(se.drawn_track()) == 3
    se.ep_track0 = 4                  # …and one before it changes nothing
    assert len(se.drawn_track()) == 5


def test_get_trajectory_reads_the_same_slice_the_map_draws():
    """The picture and the numbers must not disagree about where you have been."""
    se = walker()
    se.status, se.tick = "ok", 0
    walk(se, 7)
    se.begin_task(1)
    walk(se, 4)
    out = se.get_trajectory()
    assert out["n_total"] == 4, "the whole visit would be 11"
    assert out["points"] == [[7.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0]]
    assert [p[0] for p in se.drawn_track()] == [q[0] for q in out["points"]]
