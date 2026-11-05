"""Stage 22, new places: an office floor plan made of WALL cells. Spec: docs/specs/2026-11-05-office-design.md

A layout is fixed by its seed. Along each axis the 48-cell ring is cut into bays: a corridor 2 cells wide, then one or two
rooms with a wall on each side. A cell is open if it's in a corridor on either axis, a wall if it's on a wall line on either
axis, and room floor otherwise. So corridors form a grid that wraps around the map, and rooms sit in the blocks between them.
Every room gets a doorway (a gap 1 or 2 cells wide) in a wall it shares with a corridor, if it has one, and each wall between
two rooms gets a doorway half the time. A last pass opens a door from any room still cut off into a reachable neighbour, so
every open cell can be reached from every other. Walls block movement and show in the wall input channel; they don't block
sight, so this is an office with glass partitions.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

CORRIDOR, WALL_LINE, ROOM = 0, 1, 2  # per-axis labels
CORRIDOR_WIDTH = 2
SEED_BASE = 20_000  # so office seeds don't collide with the hill or thicket maps' streams
TRAIN_SEED, TEST_SEED = 0, 1000  # first seed of the training bank and of the held-out bank


def _axis(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n,) labels for one axis: bays of corridor, wall, room [, wall, room], wall, rolled to a random start."""
    bays = max(1, round(n / 16))
    widths = np.full(bays, n // bays)
    widths[: n % bays] += 1
    for _ in range(bays):  # move a few cells between bays, keeping each between 12 and 20
        i, j = rng.integers(0, bays, 2)
        d = int(rng.integers(0, 4))
        if i != j and widths[i] - d >= 12 and widths[j] + d <= 20:
            widths[i] -= d
            widths[j] += d
    out = []
    for w in widths:
        inner = int(w) - CORRIDOR_WIDTH
        rooms = 2 if inner >= 11 and rng.random() < 0.6 else 1
        floor = inner - (rooms + 1)  # cells of room floor after the walls
        if rooms == 1:
            sizes = [floor]
        else:
            a = int(rng.integers(max(3, floor - 8), min(floor - 3, 8) + 1))
            sizes = [a, floor - a]
        seq = [CORRIDOR] * CORRIDOR_WIDTH + [WALL_LINE]
        for k, s in enumerate(sizes):
            seq += [ROOM] * s + [WALL_LINE]
        out += seq
    assert len(out) == n, (len(out), n)
    return np.roll(np.array(out, np.int8), int(rng.integers(0, n)))


def _runs(labels: np.ndarray, value: int) -> list[np.ndarray]:
    """Index runs (wrapping) of cells with this label."""
    n = len(labels)
    start = int(np.argmax(labels != value)) if (labels != value).any() else 0
    idx = [(start + i) % n for i in range(n)]
    runs, cur = [], []
    for i in idx:
        if labels[i] == value:
            cur.append(i)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


def reachable(open_: np.ndarray) -> np.ndarray:
    """(n, n) bool: open cells reachable from the first open cell, by steps north, south, east and west on the torus."""
    n = open_.shape[0]
    seen = np.zeros_like(open_, bool)
    ys, xs = np.nonzero(open_)
    if not len(ys):
        return seen
    stack = [(int(ys[0]), int(xs[0]))]
    seen[ys[0], xs[0]] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v, u = (y + dy) % n, (x + dx) % n
            if open_[v, u] and not seen[v, u]:
                seen[v, u] = True
                stack.append((v, u))
    return seen


@lru_cache(maxsize=256)
def _layout(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(SEED_BASE + seed)
    lx, ly = _axis(n, rng), _axis(n, rng)
    X, Y = np.meshgrid(lx, ly)  # [y, x]
    wall = ((X == WALL_LINE) | (Y == WALL_LINE)) & (X != CORRIDOR) & (Y != CORRIDOR)
    rooms = [(ry, rx) for ry in _runs(ly, ROOM) for rx in _runs(lx, ROOM)]

    def side_cells(ry, rx, side):
        """The wall cells along one side of a room, and the cells just beyond them."""
        if side in ("w", "e"):
            x = (rx[0] - 1) % n if side == "w" else (rx[-1] + 1) % n
            beyond = (x - 1) % n if side == "w" else (x + 1) % n
            return [(int(y), x) for y in ry], [(int(y), beyond) for y in ry]
        y = (ry[0] - 1) % n if side == "n" else (ry[-1] + 1) % n
        beyond = (y - 1) % n if side == "n" else (y + 1) % n
        return [(y, int(x)) for x in rx], [(beyond, int(x)) for x in rx]

    def open_door(cells):
        width = min(len(cells), int(rng.integers(1, 3)))
        at = int(rng.integers(0, len(cells) - width + 1))
        for y, x in cells[at: at + width]:
            wall[y, x] = False

    for ry, rx in rooms:
        corridor_sides, room_sides = [], []
        for side in "nsew":
            cells, beyond = side_cells(ry, rx, side)
            by, bx = beyond[len(beyond) // 2]
            if lx[bx] == CORRIDOR or ly[by] == CORRIDOR:
                corridor_sides.append(cells)
            elif side in "se":  # each wall between two rooms is visited once, from its north or west room
                room_sides.append(cells)
        if corridor_sides:
            open_door(corridor_sides[int(rng.integers(0, len(corridor_sides)))])
        for cells in room_sides:
            if rng.random() < 0.5:
                open_door(cells)
    for _ in range(len(rooms)):  # repair: a door from any room still cut off into a reachable neighbour
        seen = reachable(~wall)
        if seen[~wall].all():
            break
        for ry, rx in rooms:
            if seen[ry[0], rx[0]]:
                continue
            for side in "nsew":
                cells, beyond = side_cells(ry, rx, side)
                hits = [i for i, (y, x) in enumerate(beyond) if seen[y, x]]
                if hits:
                    y, x = cells[hits[len(hits) // 2]]
                    wall[y, x] = False
                    break
            else:
                continue
            break
    assert reachable(~wall)[~wall].all(), f"office seed {seed} isn't connected"
    wall.setflags(write=False)
    return wall


def office_map(size: int, seed: int) -> np.ndarray:
    """(size, size) bool, indexed [y, x]: the wall cells of office layout `seed`."""
    return _layout(int(size), int(seed))


def office_bank(size: int, first_seed: int, count: int) -> np.ndarray:
    """(count, size, size) bool: layouts first_seed, first_seed + 1, ... A world picks one of them at each reset."""
    return np.stack([office_map(size, first_seed + i) for i in range(count)])


def ascii_map(wall: np.ndarray) -> str:
    return "\n".join("".join("#" if c else "." for c in row) for row in wall)
