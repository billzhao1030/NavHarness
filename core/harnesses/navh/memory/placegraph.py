"""The house as a graph of named places, not one big grid.

    G_house = (G_place, G_room, G_floor, E_connector)

The env keeps several storeys of occupancy, but a route is planned on one
storey's grid. So what is missing across floors is not another
map — it is the fact that connects them. A robot can know the kitchen is
upstairs, hold the upstairs layer, and still have no way to say how to get
onto it.

Nothing here needs new sensing. Every node comes from something the harness
already writes down:

    Place      a marker the agent dropped, with the views it took there
    Room       the room a place was assigned to (the marker's own name is the
               default: agents label markers "kitchen", "stairs", "front door")
    Floor      the occupancy layer the body was on when the marker was dropped
    Connector  inferred: two places marked consecutively on DIFFERENT floors
               means the robot walked between them, so something joins them

That last inference is the whole trick, and it is worth being honest about its
limit: it records that a crossing happened between two places, not where the
staircase is. That is enough to plan `place -> connector -> floor -> room` and
leave the metric leg to the path preview, the same division of labour the
rest of the harness uses.

Addresses here are labels and relations. No coordinates cross this boundary.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Place:
    label: str
    floor: int = 0
    room: str = ""
    photos: int = 0                 # how many views were banked at this place
    visits: int = 0
    last_task: int = -1

    def address(self) -> str:
        bits = [self.label]
        if self.room and self.room != self.label:
            bits.append(f"in the {self.room}")
        bits.append(f"on floor {self.floor + 1}")
        return " ".join(bits)


@dataclass
class Connector:
    """A crossing that was actually walked, between two named places."""
    a: str
    b: str
    floor_a: int
    floor_b: int
    times: int = 1

    def key(self) -> tuple:
        return tuple(sorted((self.a, self.b)))


class PlaceGraph:
    def __init__(self) -> None:
        self.places: dict[str, Place] = {}
        self.connectors: dict[tuple, Connector] = {}
        self._last: Place | None = None          # the previous place marked, for the crossing rule

    # ── building, from what the harness already sees ─────────────────────

    def note_mark(self, label: str, floor: int, *, room: str = "", photos: int = 0,
                  task: int = -1) -> Place:
        p = self.places.get(label)
        if p is None:
            p = self.places[label] = Place(label=label, floor=int(floor), room=room or label)
        p.floor = int(floor)
        p.visits += 1
        p.last_task = task
        p.photos = max(p.photos, int(photos))
        if room:
            p.room = room
        if self._last is not None and self._last.label != label and self._last.floor != p.floor:
            c = Connector(self._last.label, p.label, self._last.floor, p.floor)
            have = self.connectors.get(c.key())
            if have is None:
                self.connectors[c.key()] = c
            else:
                have.times += 1
        self._last = p
        return p

    def clear(self) -> None:
        """Forget every place and crossing: the markers they were named on are gone."""
        self.places.clear()
        self.connectors.clear()
        self._last = None

    # ── reading ──────────────────────────────────────────────────────────

    def floors(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for p in self.places.values():
            out.setdefault(p.floor, []).append(p.label)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def rooms(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self.places.values():
            out.setdefault(p.room, []).append(p.label)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def route(self, here: str, there: str) -> list[str]:
        """A coarse route as a list of place labels.

        Two kinds of edge, and they are not the same kind of fact. Within one
        storey every place is reachable from every other — that is what the
        occupancy grid is for, and the metric leg is the path preview's job, so
        this graph does not pretend to know the corridor. Between storeys only
        a crossing the robot has actually walked counts, because that is the
        one thing the grid cannot tell you.
        """
        a, b = self.places.get(here), self.places.get(there)
        if a is None or b is None:
            return []
        if a.floor == b.floor:
            return [a.label, b.label]

        adj: dict[str, set[str]] = {lbl: set() for lbl in self.places}
        for _f, labels in self.floors().items():         # same storey: fully connected
            for x in labels:
                adj[x] |= set(labels) - {x}
        for c in self.connectors.values():               # between storeys: only walked crossings
            adj.setdefault(c.a, set()).add(c.b)
            adj.setdefault(c.b, set()).add(c.a)

        seen, queue = {a.label}, [[a.label]]
        while queue:
            path = queue.pop(0)
            if path[-1] == b.label:
                return path
            for nxt in sorted(adj.get(path[-1], ())):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append([*path, nxt])
        return []

    def candidates_on(self, floor: int) -> list[Place]:
        """Where a coarse-to-fine wake-up should look first on a given storey:
        the places that banked the most views, since those are the ones a photo
        match can actually confirm."""
        return sorted((p for p in self.places.values() if p.floor == floor),
                      key=lambda p: (-p.photos, -p.visits, p.label))

    def render(self, *, max_lines: int = 24) -> str:
        """The graph as prose, for the context. Labels and relations only."""
        if not self.places:
            return ""
        out = ["What you know about the shape of this house:"]
        for f, labels in self.floors().items():
            out.append(f"  floor {f + 1}: " + ", ".join(labels))
        for c in self.connectors.values():
            out.append(f"  you have walked between {c.a} (floor {c.floor_a + 1}) and "
                       f"{c.b} (floor {c.floor_b + 1}) — that is the way between those storeys")
        return "\n".join(out[:max_lines])

    # ── it outlives the process, like the map does ───────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"places": [asdict(p) for p in self.places.values()],
             "connectors": [asdict(c) for c in self.connectors.values()]},
            indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> PlaceGraph:
        g = cls()
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return g
        for p in d.get("places") or []:
            try:
                g.places[p["label"]] = Place(**p)
            except (TypeError, KeyError):
                continue
        for c in d.get("connectors") or []:
            try:
                cc = Connector(**c)
                g.connectors[cc.key()] = cc
            except (TypeError, KeyError):
                continue
        return g
