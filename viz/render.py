"""Render world states to RGB frames and MP4 clips."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image, ImageDraw

from agents.policies import forget_after, pair_act, pair_memory

from simulations import predprey as pp

CELL = 10
HEADER = 32
SOIL = (48, 36, 24)
GRASS = (46, 150, 60)
PREY = (235, 235, 235)
PRED = (220, 45, 45)
TOUGH_GRASS = (18, 70, 28)
WALL_RGB = (128, 128, 128)
BURROW_RGB = (92, 54, 22)
SLOW_PREY, FAST_PREY = (120, 120, 120), (255, 255, 255)
SLOW_PRED, FAST_PRED = (110, 25, 25), (255, 70, 70)
HEADER_BG = (16, 16, 16)
HEADER_TEXT = (230, 230, 230)


def _lerp(a, b, f):
    return np.array(a) * (1 - f) + np.array(b) * f


def _canvas(cfg, img, text) -> np.ndarray:
    side = cfg.size * CELL
    canvas = Image.new("RGB", (side, side + HEADER), HEADER_BG)
    canvas.paste(Image.fromarray(img), (0, HEADER))
    ImageDraw.Draw(canvas).text((8, 10), text, fill=HEADER_TEXT)
    return np.asarray(canvas)


def render_evo_frame(cfg, state, label: str = "") -> np.ndarray:
    """Plants darken with toughness; animals brighten with speed."""
    plant = np.asarray(state.plant)
    tough = np.asarray(state.plant_genes)[..., 1]
    rgb = np.where(plant[..., None], _lerp(GRASS, TOUGH_GRASS, tough[..., None]), np.array(SOIL))
    bitten = np.asarray(state.bitten)[..., None]
    rgb = np.where(bitten, _lerp(rgb, SOIL, 0.5), rgb)  # regrowing after a bite
    terrain = np.asarray(state.terrain)[..., None]
    rgb = np.where(terrain == 1, np.array(WALL_RGB), np.where(terrain == 2, np.array(BURROW_RGB), rgb))
    img = np.repeat(np.repeat(rgb.astype(np.uint8), CELL, axis=0), CELL, axis=1)

    def paint(sp, slow, fast, inset: int) -> float:
        alive = np.asarray(sp.alive)
        speeds = np.asarray(sp.genes)[alive, 0]
        for (x, y), s in zip(np.asarray(sp.pos)[alive], speeds):
            img[y * CELL + inset:(y + 1) * CELL - inset, x * CELL + inset:(x + 1) * CELL - inset] = _lerp(slow, fast, s)
        return float(speeds.mean()) if speeds.size else 0.0

    prey_speed = paint(state.prey, SLOW_PREY, FAST_PREY, 2)
    pred_speed = paint(state.pred, SLOW_PRED, FAST_PRED, 1)
    n_prey = int(np.asarray(state.prey.alive).sum())
    n_pred = int(np.asarray(state.pred.alive).sum())
    mean_tough = float(tough[plant].mean()) if plant.any() else 0.0
    text = (f"t={int(state.t):3d}  prey {n_prey:2d} spd {prey_speed:.2f}  pred {n_pred:2d} spd {pred_speed:.2f}"
            f"  plant tough {mean_tough:.2f}  {label}")
    return _canvas(cfg, img, text)


def render_frame(cfg: pp.WorldConfig, state, label: str = "") -> np.ndarray:
    if hasattr(state, "plant"):
        return render_evo_frame(cfg, state, label)
    grass = np.asarray(state.grass)
    regrown = 1.0 - np.clip(grass / cfg.regrow_steps, 0.0, 1.0)
    frac = np.where(grass == 0, 1.0, 0.6 * regrown)[..., None]
    rgb = (np.array(SOIL) * (1 - frac) + np.array(GRASS) * frac).astype(np.uint8)
    img = np.repeat(np.repeat(rgb, CELL, axis=0), CELL, axis=1)

    def paint(sp: pp.Species, colour, inset: int) -> None:
        for x, y in np.asarray(sp.pos)[np.asarray(sp.alive)]:
            img[y * CELL + inset:(y + 1) * CELL - inset, x * CELL + inset:(x + 1) * CELL - inset] = colour

    paint(state.prey, PREY, 2)
    paint(state.pred, PRED, 1)

    n_prey = int(np.asarray(state.prey.alive).sum())
    n_pred = int(np.asarray(state.pred.alive).sum())
    return _canvas(cfg, img, f"t={int(state.t):3d}   prey {n_prey:2d}   predators {n_pred:2d}   {label}")


def rollout_states(cfg, prey_policy, pred_policy, key, max_frames: int, world=None, init_state=None) -> list:
    """States of one episode, from `init_state` (or a fresh reset) until it ends."""
    world = world or pp

    @jax.jit
    def advance(state, mem, k):
        kp, kq = jax.random.split(k)
        prey_obs, pred_obs = world.observe(cfg, state)
        prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
        res = world.advance(cfg, jax.random.fold_in(k, 1), state, prey_a, pred_a)
        return res.state, forget_after(mem, res), res.done

    k_reset, key = jax.random.split(key)
    state = world.reset(cfg, k_reset) if init_state is None else init_state
    states = [jax.device_get(state)]
    mem = pair_memory(prey_policy, pred_policy, state)
    for k in jax.random.split(key, max(max_frames - 1, 0)):
        state, mem, done = advance(state, mem, k)
        states.append(jax.device_get(state))
        if bool(done):
            break
    return states


def write_clip(cfg, prey_policy, pred_policy, key, path, max_frames=None, fps=15, label="",
               world=None, init_state=None) -> int:
    states = rollout_states(cfg, prey_policy, pred_policy, key, max_frames or cfg.max_steps, world, init_state)
    frames = [render_frame(cfg, s, label) for s in states]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=7)
    return len(frames)
