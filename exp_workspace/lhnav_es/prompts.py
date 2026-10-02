"""lhnav_es briefing — the single-session system prompts for this arm's
lines, each followed by its SLAM instrument addendum.

``build_briefing(..., task="goat", n_goals=N)`` renders GOAT_SYSTEM_PROMPT +
the objnav instrument addendum (markers that outlive sub-goals); the other
task words render the R2R/RxR ("nav"), ObjectNav ("objnav") and HM-EQA
("eqa") prompts, in the two-tool (observe/step) or the singlestep face. The
runner renders this briefing for every episode; the NavHarness seats run on
their own standing brief (core/harnesses/navh/prompts.py).

The nav prompt treats the instruction as a sequence of loose checkpoints to
pass in order (between checkpoints the path is free), and asks for a stop
well inside the 3 m success radius rather than at its edge.
"""
from __future__ import annotations

_OBS_NOTE_SEP = ""
_STEP_NOTE_SEP = ""
_STEP_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide where the instruction "
    "wants you to go next, move, look again."
)

BARE_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).{obs_note}
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.{step_note}

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
- The instruction is a SCRIPT, not a hint: it describes a sequence of \
loose checkpoints — each room entered, turn taken, landmark passed. Pass \
through them in order, completing one before heading for the next; between \
checkpoints the exact path is yours to choose. Skipping checkpoints — a \
shortcut, or wandering that happens to reach the goal — scores worse than \
visiting them in order. If you get lost, backtrack to the last checkpoint \
you completed; do not switch to hunting the endpoint.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of \
the instruction's endpoint — but treat 3 meters as the hard limit, NOT the \
target. Keep approaching until the described final object or area is \
directly in front of you, ideally within 1 meter; if the endpoint is an \
area (a room, a doorway, "near the tub and sink"), walk to its center \
before stopping, not its edge. STOP is permanent — issue it only there.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

# Map-v2 instrument addendum: no frontier layer; the map is floor-layered and
# its coordinate window is fixed between calls; look_around, mark,
# preview_path and get_map floor/zoom are all free.
SLAM_INSTRUMENT_ADDENDUM_V2 = """

You additionally have six FREE instrument tools backed by an onboard SLAM
system (no step cost, no budget, callable any time):

- get_pose(): your estimated position (x, z, meters) and heading (yaw_deg) in
  a fixed frame anchored at your start pose.
- get_map(): a top-down occupancy map built automatically as you move (it
  already contains an initial 360-degree scan of your start location). Its
  coordinate grid stays fixed between calls, and it shows your current
  floor's layer — a "floor k/n" tag marks the storey when a scene has
  several; pass floor=k to view another storey, or zoom=true to get both
  the full map and a fixed 8-meter close-up around you. Markers you drop
  with mark() appear on it.
- get_trajectory(): the path you have walked on this task,
  oldest first (the map keeps the whole visit).
- look_around(): turn a full circle in place and get four views 90 degrees
  apart (front / right / back / left of your current heading), ending faced
  where you started. A pure look, like observe(): it does not change the
  map — only walking builds the map.
- mark(label): drop a named marker at your position (re-marking a label
  moves it) — it shows on the map, so you can pin places the instruction
  mentions instead of memorizing them.
- preview_path(x, z): a read-only route preview over the map's walked-free
  space to point (x, z), drawn on the map with its length in meters —
  nothing moves.

These solve "where am I / where have I been / what is around me" — use them
instead of guessing from visual memory, and use them to STAY ON the
instruction's route (verify a turn you just made, recognize a room you were
told to enter) — not to hunt for the endpoint by open exploration. They are
SLAM estimates, not ground truth, and can drift slightly."""


# ── HM-EQA profile ──
# The observe/step/answer toolface with the tilt fills (EQA_TILT_* below),
# followed by the map-v2 instrument addendum reworded for question answering.
EQA_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). {camera_sentence} You interact only through these \
tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 1 = move forward \
0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 degrees{tilt_actions}. \
There is no stop action.
- answer(letter): permanently END the episode by answering the question \
with "A", "B", "C" or "D".

Your task is embodied question answering: explore the building until you \
can answer this multiple-choice question about it:

"{question}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions. If it runs out you can \
still observe and answer from where you stand.
- You succeed only if answer() gives the correct letter. answer() is \
permanent — call it once you have seen enough evidence to be confident, \
and always answer before ending: an episode without answer() scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""

EQA_TILT_CAMERA_SENTENCE = "Its camera starts tilted 30 degrees downward."
EQA_FIXED_CAMERA_SENTENCE = "Its camera is tilted 30 degrees downward."
EQA_TILT_ACTIONS_CLAUSE = (
    ", 4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 "
    "degrees (tilt changes the camera pitch only, not your position or "
    "heading)"
)
EQA_TILT_RULE = (
    "\n- Tilt changes persist until you change them again; check your "
    "camera pitch before interpreting a view."
)

SLAM_INSTRUMENT_ADDENDUM_V2_EQA = """

You additionally have six FREE instrument tools backed by an onboard SLAM
system (no step cost, no budget, callable any time):

- get_pose(): your estimated position (x, z, meters) and heading (yaw_deg) in
  a fixed frame anchored at your start pose.
- get_map(): a top-down occupancy map built automatically as you move (it
  already contains an initial 360-degree scan of your start location). Its
  coordinate grid stays fixed between calls, and it shows your current
  floor's layer — a "floor k/n" tag marks the storey when a building has
  several; pass floor=k to view another storey, or zoom=true to get both
  the full map and a fixed 8-meter close-up around you. Markers you drop
  with mark() appear on it.
- get_trajectory(): the path you have walked on this task,
  oldest first (the map keeps the whole visit).
- look_around(): turn a full circle in place and get four views 90 degrees
  apart (front / right / back / left of your current heading, at your
  current camera pitch), ending faced where you started. A pure look, like
  observe(): it does not change the map — only walking builds the map.
- mark(label): drop a named marker at your position (re-marking a label
  moves it) — it shows on the map, so you can pin rooms and objects that
  matter to the question instead of memorizing them.
- preview_path(x, z): a read-only route preview over the map's walked-free
  space to point (x, z), drawn on the map with its length in meters —
  nothing moves.

These solve "where am I / where have I been / what is around me" — use them
to explore SYSTEMATICALLY instead of guessing from visual memory: read the
map for rooms and doorways you have not entered yet, pin the places you
checked, and preview a route before walking it, so you cover the building
without circling. They are SLAM estimates, not ground truth, and can drift
slightly."""


# ── ObjectNav profile ──
# Two-tool observe/step skeleton aligned with BARE_SYSTEM_PROMPT; the task
# section and rules use the ObjectNav wording (search rule, "0.5 meters"
# success rule, "any instance counts"), on the STANDARD body (the R2R/RxR
# action space: 15° turns, no tilt).
OBJNAV_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have found the target), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Your task is object-goal navigation: no route is given — search the \
building until you find the target object, walk up to it, and stop there.

Target object: "{goal}"

Rules:
- Alternate observing and stepping: look, decide where to search next, \
move, look again.
- Explore efficiently: sweep toward where a "{goal}" is most likely to \
be, and avoid re-walking areas you have already ruled out.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 0.5 meters \
of a "{goal}". Any instance counts. STOP is permanent — issue it only \
when you can see the target and are standing right next to it.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look \
around when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

SLAM_INSTRUMENT_ADDENDUM_V2_OBJNAV = """

You additionally have six FREE instrument tools backed by an onboard SLAM
system (no step cost, no budget, callable any time):

- get_pose(): your estimated position (x, z, meters) and heading (yaw_deg) in
  a fixed frame anchored at your start pose.
- get_map(): a top-down occupancy map built automatically as you move (it
  already contains an initial 360-degree scan of your start location). Its
  coordinate grid stays fixed between calls, and it shows your current
  floor's layer — a "floor k/n" tag marks the storey when a building has
  several; pass floor=k to view another storey, or zoom=true to get both
  the full map and a fixed 8-meter close-up around you. Markers you drop
  with mark() appear on it.
- get_trajectory(): the path you have walked on this task,
  oldest first (the map keeps the whole visit).
- look_around(): turn a full circle in place and get four views 90 degrees
  apart (front / right / back / left of your current heading), ending faced
  where you started. A pure look, like observe(): it does not change the
  map — only walking builds the map.
- mark(label): drop a named marker at your position (re-marking a label
  moves it) — it shows on the map, so you can pin rooms you have checked
  and spots worth returning to instead of memorizing them.
- preview_path(x, z): a read-only route preview over the map's walked-free
  space to point (x, z), drawn on the map with its length in meters —
  nothing moves.

These solve "where am I / where have I been / what is around me" — use them
to search SYSTEMATICALLY instead of guessing from visual memory: read the
map for rooms and doorways you have not entered yet, pin the places you
ruled out, and preview a route before walking it, so you cover the building
without circling. They are SLAM estimates, not ground truth, and can drift
slightly."""


GOAT_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the whole episode — normally you never need it), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees, 6 = \
SUBTASK_STOP (declares the CURRENT sub-goal reached).
- get_goal(): re-read the current sub-goal; when it is given as a \
photograph, this returns the photograph.

Your task is lifelong multi-object navigation (GOAT-Bench): this episode has \
{n_goals} sub-goals in this same building, to be reached ONE AT A TIME, in \
order. Each sub-goal is given in one of three ways — an object category \
(any instance counts), a language description of one specific object, or a \
photo of one specific object. You only ever see the CURRENT sub-goal; the \
next one is announced after you close the current one. Targets can repeat \
later in the episode, so remember where things are.

Sub-goal 1 of {n_goals}: {instruction}

Rules:
- Alternate observing and stepping: look, decide where to search next, \
move, look again.
- When you are standing right next to the current target (within 0.25 m), \
issue action 6 (SUBTASK_STOP) to close it. The step result then announces \
the next sub-goal — read it before moving on; when it is a photo, the photo \
is attached to that result (and get_goal() returns it again any time).
- Closing a sub-goal is permanent: you are scored on where you stood when \
you closed it, whether or not you had found it. Closing the last sub-goal \
ends the episode. Action 0 (STOP) also ends the whole episode — you should \
not need it.
- You have a budget of {budget} movement actions for the whole episode.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until the episode ends; nobody can answer questions.
"""

SLAM_INSTRUMENT_ADDENDUM_V2_GOAT = SLAM_INSTRUMENT_ADDENDUM_V2_OBJNAV.replace(
    "so you can pin rooms you have checked\n  and spots worth returning to instead of memorizing them.",
    "so you can pin rooms you have checked\n  and objects you passed — a later sub-goal may ask for them — instead of\n  memorizing them.",
).replace(
    "They are SLAM estimates, not ground truth, and can drift\nslightly.",
    "The map and the markers persist across sub-goals: by the second or third\n"
    "sub-goal the map already holds much of the building. They are SLAM\n"
    "estimates, not ground truth, and can drift slightly.",
)


# ── Singlestep-face briefings: one step tool that returns the view after its
# actions. MIP-aligned bases (not this arm's nav briefing) + this arm's
# instrument addenda with the look_around analogy swapped observe() → step([]). ──
BARE_SYSTEM_PROMPT_SS = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through this tool:

- step(actions): execute movement actions in order, then return the \
robot's forward-facing camera view (an RGB image) after the last action. \
0 = STOP (permanently ends the episode — declares you have reached the \
goal), 1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn right \
15 degrees. Calling step([]) with no actions returns the current view \
without moving.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Call step([]) once at the start to see where you are.
- Every step call shows you the view after your actions: study it, decide \
where the instruction wants you to go next, and issue the next step. You \
never need a separate look call.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

OBJNAV_SYSTEM_PROMPT_SS = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through this tool:

- step(actions): execute movement actions in order, then return the \
robot's forward-facing camera view (an RGB image) after the last action. \
0 = STOP (permanently ends the episode — declares you have found the \
target), 1 = move forward 0.25 m, 2 = turn left 15 degrees, 3 = turn \
right 15 degrees. Calling step([]) with no actions returns the current view \
without moving.

Your task is object-goal navigation: no route is given — search the \
building until you find the target object, walk up to it, and stop there.

Target object: "{goal}"

Rules:
- Call step([]) once at the start to see where you are.
- Every step call shows you the view after your actions: study it, decide \
where to search next, and issue the next step. You never need a separate \
look call.
- Explore efficiently: sweep toward where a "{goal}" is most likely to \
be, and avoid re-walking areas you have already ruled out.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 0.5 meters \
of a "{goal}". Any instance counts. STOP is permanent — issue it only \
when you can see the target and are standing right next to it.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look \
around when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

EQA_SYSTEM_PROMPT_SS = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). {camera_sentence} You interact only through these \
tools:

- step(actions): execute movement actions in order, then return the \
robot's forward-facing camera view (an RGB image) after the last action. \
1 = move forward 0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 \
degrees{tilt_actions}. There is no stop action. Calling step([]) with no \
actions returns the current view without moving.
- answer(letter): permanently END the episode by answering the question \
with "A", "B", "C" or "D".

Your task is embodied question answering: explore the building until you \
can answer this multiple-choice question about it:

"{question}"

Rules:
- Call step([]) once at the start to see where you are.
- Every step call shows you the view after your actions: study it, decide \
where to go to find the evidence the question needs, and issue the next \
step. You never need a separate look call.
- You have a budget of {budget} movement actions. If it runs out you can \
still look (step([])) and answer from where you stand.
- You succeed only if answer() gives the correct letter. answer() is \
permanent — call it once you have seen enough evidence to be confident, \
and always answer before ending: an episode without answer() scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""


def build_briefing(instruction: str, step_budget: int,
                   task: str = "nav", tilt: bool = True,
                   face: str = "twotool", n_goals: int = 1) -> str:
    """``task`` "goat" (``instruction`` is the FIRST sub-goal's statement,
    ``n_goals`` the sequence length; two-tool face only),
    "nav" (default — R2R/RxR),
    "eqa" (HM-EQA: ``instruction`` is the formatted multiple-choice
    question; ``tilt`` mirrors the cell's tilt_actions knob), or
    "objnav" (ObjectNav: ``instruction`` is the goal-category display
    text; no tilt — the STANDARD body). ``face``
    "twotool" (default) or "singlestep" — the singlestep bases are the
    MIP-aligned *_SS templates, not this arm's nav briefing."""
    if task == "goat":
        if face != "twotool":
            raise ValueError("the GOAT surface has no singlestep face (lhnav_es)")
        base = GOAT_SYSTEM_PROMPT.format(instruction=instruction, n_goals=int(n_goals),
                                         budget=step_budget)
        return base + SLAM_INSTRUMENT_ADDENDUM_V2_GOAT
    if face == "singlestep":
        if task == "objnav":
            base = OBJNAV_SYSTEM_PROMPT_SS.format(goal=instruction,
                                                  budget=step_budget)
            add = SLAM_INSTRUMENT_ADDENDUM_V2_OBJNAV
        elif task == "eqa":
            fills = ({"camera_sentence": EQA_TILT_CAMERA_SENTENCE,
                      "tilt_actions": EQA_TILT_ACTIONS_CLAUSE,
                      "tilt_rule": EQA_TILT_RULE}
                     if tilt else
                     {"camera_sentence": EQA_FIXED_CAMERA_SENTENCE,
                      "tilt_actions": "", "tilt_rule": ""})
            base = EQA_SYSTEM_PROMPT_SS.format(question=instruction,
                                               budget=step_budget, **fills)
            add = SLAM_INSTRUMENT_ADDENDUM_V2_EQA
        else:
            base = BARE_SYSTEM_PROMPT_SS.format(instruction=instruction,
                                                budget=step_budget)
            add = SLAM_INSTRUMENT_ADDENDUM_V2
        # singlestep face has no observe(): the addendum's look_around
        # analogy swaps to the step([]) peek (bridge does the same). The
        # phrase line-wraps in the eqa/objnav addenda — both forms handled,
        # wrap preserved.
        add = (add.replace("like observe()", "like step([])")
                  .replace("like\nobserve()", "like\nstep([])")
                  .replace("like\n  observe()", "like\n  step([])"))
        return base + add
    if task == "objnav":
        base = OBJNAV_SYSTEM_PROMPT.format(goal=instruction, budget=step_budget)
        return base + SLAM_INSTRUMENT_ADDENDUM_V2_OBJNAV
    if task == "eqa":
        fills = ({"camera_sentence": EQA_TILT_CAMERA_SENTENCE,
                  "tilt_actions": EQA_TILT_ACTIONS_CLAUSE,
                  "tilt_rule": EQA_TILT_RULE}
                 if tilt else
                 {"camera_sentence": EQA_FIXED_CAMERA_SENTENCE,
                  "tilt_actions": "", "tilt_rule": ""})
        base = EQA_SYSTEM_PROMPT.format(question=instruction, budget=step_budget, **fills)
        return base + SLAM_INSTRUMENT_ADDENDUM_V2_EQA
    base = BARE_SYSTEM_PROMPT.format(
        instruction=instruction, budget=step_budget,
        obs_note=_OBS_NOTE_SEP, step_note=_STEP_NOTE_SEP,
        loop_rule=_STEP_LOOP_SEP,
    )
    return base + SLAM_INSTRUMENT_ADDENDUM_V2
