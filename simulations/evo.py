"""Evolving Crucible: the predator-prey world with heritable bodies and plants.

Animals inherit speed, sight and breeding genes from their parent with
mutation; plants spread by seeding and inherit spread and toughness. Each
species still has one learned brain, which sees the animal's own genes.
Gene pools pass between worlds when a world resets (see `batch_step`).
Positions are (x, y); 2-D maps are indexed [y, x].
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from simulations import predprey as pp
from simulations.predprey import NUM_ACTIONS, WorldConfig  # noqa: F401  (re-exported)

OBS_CHANNELS = 11  # plant, toughness, prey, predators | own energy, speed, sight, breed | wall, burrow | own size
PREY_CHANNEL = 2  # where prey show up in an observation
WALL, BURROW = 1, 2  # State.terrain codes; 0 is open ground
ANIMAL_GENES = 6  # speed, sight, breed, size, brain, diet
PLANT_GENES = 2  # spread, toughness
# (dx, dy) to the four neighbours a plant can seed: north, south, east, west.
NEIGHBOURS = ((0, -1), (0, 1), (1, 0), (-1, 0))
FOUNDER_SIZE = 0.5  # size given to animals saved or written without one
POOL_SAMPLE = 4096  # plants drawn from at reset; genomes kept per kind in a saved snapshot


@dataclass(frozen=True)
class EvoConfig(WorldConfig):
    max_prey: int = 96  # trained grazers filled 64 slots, and the cap (not food) then drove selection
    pred_start_energy: float = 60.0  # 30 starved every predator within ~30 steps in pp-001
    pred_base_cost: float = 0.5  # predators idle cheaply; at 1.0 per step pp-evo-001's starved
    catch_energy: float = 25.0  # a prey is a big meal; 15 needed a catch every ~14 steps
    breed_span: float = 0.8  # threshold repro * (0.5 + span * b) stays below the energy cap
    gene_sigma: float = 0.05
    founder_noise: float = 0.05
    founder_animal: tuple = (0.2, 0.5, 0.5, 0.5, 0.5, 0.5)  # speed, sight, breed, size, brain, diet
    founder_plant: tuple = (0.5, 0.2)
    speed_cost: float = 1.0
    pred_speed_cost: float = -1.0  # what speed costs hunters; < 0 = speed_cost (Stage 10e)
    sight_cost: float = 0.5
    seed_rate: float = 0.03  # 0.2 left food plentiful and prey stuck at the slot cap
    tough_seed_cost: float = 0.8
    die_rate: float = 0.02
    plant_base_death: float = 0.3  # share of die_rate every plant pays; without it spread 0 was immortal
    bite_regrow: float = 0.1  # chance per step a bitten plant regrows; until then it cannot feed or seed
    birth_cost: float = 8.0  # energy lost per birth; without it breeding thresholds collapsed
    germinate_rate: float = 0.0005
    walls: int = 0  # straight wall segments drawn at each reset
    wall_len: int = 6
    burrows: int = 0  # single cells prey can enter and predators cannot
    day_len: int = 0  # steps per day (0 = always day); at night every animal sees only its neighbours
    night_frac: float = 0.3  # the last part of each day that is night
    season_len: int = 0  # steps per season (0 = none); the map's halves take turns being in season
    winter_len: int = 0  # steps per half-year (0 = none); the whole map alternates summer and winter (Stage 9c)
    winter_value: float = 0.25  # what a plant feeds in winter, against 1 in summer
    climate: str = ""  # the viewer's colours for this continent (Stage 11a)
    offseason_death: float = 3.0  # plants out of season die this many times faster and cannot seed
    sizes: bool = False  # the size gene matters: tough plants, burrows, catches and upkeep
    burrow_size: float = 0.5  # only animals smaller than this fit into a burrow
    catch_margin: float = 0.25  # catch_scale 0: a predator catches prey up to its own size plus this
    catch_scale: float = 0.0  # > 0: a catch succeeds with chance sigmoid((pred size - prey size) / scale)
    size_catch: bool = True  # False: size plays no part in predation (Stage 5a v3)
    plant_sigma: float = -1.0  # mutation of plant genes; < 0 = gene_sigma
    plant_kinds: bool = False  # founder plants are soft grass (toughness 0) or tough shrubs (1), half each
    kinds_persist: bool = False  # bites never kill, seeding ignores toughness, germination draws founders (v5)
    size_cost: float = 0.5  # energy per step for size 1
    size_tradeoff: bool = False  # small grazers also digest soft plants better: no size is best at both (v4)
    mating: bool = False  # breeding needs a mate nearby; the child mixes both parents' genes
    mate_radius: int = 2  # wrapped Chebyshev cells
    mate_tolerance: float = 1.0  # mates differ in size by at most this (1 = any mate will do)
    pack_size: int = 0  # >= 2: a prey is eaten only when this many predators within 1 cell hunt it (Stage 6a)
    flank: bool = False  # with packs: a prey is eaten only when its hunters stand on opposite sides of it (Stage 6c)
    signals: bool = False  # 10 actions (each move with the signal off or on); own species sees last step's signals
    signal_cost: float = 0.05  # energy per signal
    brains: bool = False  # the brain gene sets how many hidden units an animal's brain uses, and costs energy (Stage 7a)
    brain_cost: float = 0.5  # energy per step for brain gene 1
    brain_spread: bool = False  # founders draw the brain gene uniformly from [0, 1], so every brain width is trained
    hills: bool = False  # cells have a fixed height: climbing costs energy, height adds sight, hills block sight (Stage 8a)
    hill_seed: int = 0  # which height map
    climb_cost: float = 4.0  # energy per unit of height climbed (heights run 0 to 1)
    hill_sight: float = 2.0  # extra sight radius by day on a peak, scaled down to 0 at mid height and below
    eye_height: float = 0.05  # how far above its cell an animal sees from
    move_cost: float = 0.0  # energy per cell moved, so waiting is cheaper than chasing (Stage 8b)
    thickets: float = 0.0  # > 0: this share of cells is a thicket; an animal in one is seen only from the next cell (Stage 8b)
    thicket_seed: int = 0  # which thicket map
    cover_hides_pred: bool = True  # False (evaluation): thickets no longer hide predators
    cover_hides_prey: bool = True  # False (evaluation): thickets no longer hide prey
    fickle: bool = False  # one plant kind feeds and the other poisons; which is hidden and flips at random (Stage 7c)
    flip_rate: float = 1 / 150  # chance per step that the good kind flips
    poison: float = 3.0  # energy lost for a bite of the bad kind
    omnivore: bool = False  # hunters can eat plants; their diet gene splits what they get from meat and plants (Stage 9a)
    plant_digest: float = 1.0  # how much of a grazer's plant meal a hunter with diet 0 gets
    diet_fixed: bool = False  # the control: every hunter gets 0.5 of each, whatever its gene says
    stealth_radius: int = 0  # > 0: a grazer sees a hunter only within this many cells (Stage 10f)
    fat_sense: bool = False  # animals can see how much energy visible prey hold (Stage 5b)
    stored_meals: bool = False  # a catch is worth what the prey stored, times meal_share, not a fixed catch_energy (Stage 5b)
    meal_share: float = 1.25  # meant to match catch_energy 25 with prey at about 20; prey eaten held about 9.5, so about 12 (5b v1)
    brain_tax: float = 0.0  # an animal with brain gene g keeps 1 - brain_tax * g of every meal, so rich and poor pay the same share (Stage 7b)


@dataclass(frozen=True)
class SeasonsConfig(EvoConfig):
    """Stage 4: night hides what is not next to you, and food moves between the map's halves."""
    day_len: int = 60
    season_len: int = 150


@dataclass(frozen=True)
class TerrainConfig(EvoConfig):
    """Stage 3a: walls make corners for predators; burrows are refuges without food for prey."""
    walls: int = 24
    burrows: int = 16


@dataclass(frozen=True)
class PackConfig(EvoConfig):
    """Stage 6a: a lone predator cannot catch anything; two together can."""
    pack_size: int = 2


@dataclass(frozen=True)
class FlankConfig(PackConfig):
    """Stage 6c: two hunters must close in from opposite sides, which two chasers seldom do by chance."""
    flank: bool = True


@dataclass(frozen=True)
class FlankSignalConfig(FlankConfig):
    signals: bool = True


@dataclass(frozen=True)
class PackBrainsConfig(PackConfig):
    """Stage 7a: hunting in pairs, where learning adds little, with brain size evolving at a cost."""
    brains: bool = True
    brain_spread: bool = True


@dataclass(frozen=True)
class FlankBrainsConfig(FlankConfig):
    """Stage 7a: flanking, where learning beats a simple chase, with brain size evolving at a cost."""
    brains: bool = True
    brain_spread: bool = True


@dataclass(frozen=True)
class PackTaxConfig(PackBrainsConfig):
    """Stage 7b: pack_brains, but the brain costs a share of each meal, not energy per step."""
    brain_cost: float = 0.0
    brain_tax: float = 0.3


@dataclass(frozen=True)
class FlankTaxConfig(FlankBrainsConfig):
    """Stage 7b: flank_brains, but the brain costs a share of each meal, not energy per step."""
    brain_cost: float = 0.0
    brain_tax: float = 0.3


@dataclass(frozen=True)
class HillsConfig(EvoConfig):
    """Stage 8a: a fixed hilly height map. Climbing costs energy, high ground sees further and
    hills hide what is behind them, from both species."""
    hills: bool = True


@dataclass(frozen=True)
class FlankHillsConfig(FlankConfig):
    """Stage 8a: flanking among hills, where a hunter can close in unseen."""
    hills: bool = True


@dataclass(frozen=True)
class AmbushConfig(EvoConfig):
    """Stage 8b, from a viewer's suggestion: moving costs energy and thickets hide whoever is inside,
    so a predator could wait in cover instead of chasing."""
    move_cost: float = 0.5
    thickets: float = 0.25


@dataclass(frozen=True)
class FickleConfig(EvoConfig):
    """Stage 7c: grass or shrubs feed and the other poisons. Nothing shows which, and it flips at random,
    so an animal can only find out by tasting and remembering during its own life."""
    fickle: bool = True
    plant_kinds: bool = True
    kinds_persist: bool = True
    plant_sigma: float = 0.0


@dataclass(frozen=True)
class FatPreyConfig(EvoConfig):
    """Stage 5b, from a viewer's comment: a predator's meal is worth what the grazer stored, and hunters can see who is fat."""
    fat_sense: bool = True
    stored_meals: bool = True


@dataclass(frozen=True)
class LastingConfig(EvoConfig):
    """Stage 10b: plants spread fast enough for hand-written animals to last (stage 10a)."""
    seed_rate: float = 0.2


@dataclass(frozen=True)
class LastingM100Config(LastingConfig):
    """Stage 10c: bigger meals, with the energy cap raised so they aren't clipped."""
    catch_energy: float = 100.0
    pred_energy_cap: float = 200.0


@dataclass(frozen=True)
class LastingM250Config(LastingConfig):
    catch_energy: float = 250.0
    pred_energy_cap: float = 500.0


@dataclass(frozen=True)
class LastingAmbushConfig(AmbushConfig):
    """Stage 10d: stage 8b's walking costs and thickets, with plants that last, so dodging costs grazers something."""
    seed_rate: float = 0.2


@dataclass(frozen=True)
class LastingAmbushM100Config(LastingAmbushConfig):
    catch_energy: float = 100.0
    pred_energy_cap: float = 200.0


@dataclass(frozen=True)
class LastingSprintConfig(LastingConfig):
    """Stage 10e: speed costs hunters nothing, so they can evolve to be built for the chase."""
    pred_speed_cost: float = 0.0


@dataclass(frozen=True)
class LastingSprintM100Config(LastingSprintConfig):
    catch_energy: float = 100.0
    pred_energy_cap: float = 200.0


@dataclass(frozen=True)
class LastingStealthConfig(LastingConfig):
    """Stage 10f: hunters are camouflaged, so a grazer sees one only from the next cell."""
    stealth_radius: int = 1


@dataclass(frozen=True)
class LastingStealthM100Config(LastingStealthConfig):
    catch_energy: float = 100.0
    pred_energy_cap: float = 200.0


@dataclass(frozen=True)
class LastingRoomConfig(LastingStealthConfig):
    """Stage 10g: three times the slots, so the slot caps don't set the numbers."""
    max_prey: int = 288
    max_pred: int = 96


@dataclass(frozen=True)
class ContinentConfig(LastingStealthConfig):
    """Stage 11a: 256 x 256 at the density of the 48 x 48 world (96 and 32 at most), with camouflage and plants that last."""
    size: int = 256
    max_prey: int = 2730
    max_pred: int = 910
    init_prey: int = 682
    init_pred: int = 227
    climate: str = "meadow"


@dataclass(frozen=True)
class ForestConfig(ContinentConfig):
    """Thickets hide whoever is inside, and walking costs energy (stage 8b's rules)."""
    move_cost: float = 0.5
    thickets: float = 0.25
    climate: str = "forest"


@dataclass(frozen=True)
class TundraConfig(ContinentConfig):
    """Half of every 120 steps is winter, when plants feed a quarter as much (stage 9c's rule)."""
    winter_len: int = 60
    climate: str = "tundra"


@dataclass(frozen=True)
class OmniConfig(EvoConfig):
    """Stage 9a: hunters keep their diet gene's share of every catch, and the rest of a grazer's share of plants."""
    omnivore: bool = True


@dataclass(frozen=True)
class OmniToughConfig(OmniConfig):
    """Stage 9a: the same, with a gut built badly for plants."""
    plant_digest: float = 0.3


@dataclass(frozen=True)
class OmniD03Config(OmniConfig):
    """Stage 9b's ladder of guts: 0.03, 0.08 and 0.20."""
    plant_digest: float = 0.03


@dataclass(frozen=True)
class OmniD08Config(OmniConfig):
    plant_digest: float = 0.08


@dataclass(frozen=True)
class OmniD20Config(OmniConfig):
    plant_digest: float = 0.2


@dataclass(frozen=True)
class OmniD11Config(OmniConfig):
    """Stage 9d: between the guts where hunters stay on meat (0.08) and mostly leave it (0.14)."""
    plant_digest: float = 0.11


@dataclass(frozen=True)
class OmniD11LeanConfig(OmniD11Config):
    """Stage 9e: the same gut, without the rich field of plants at the start."""
    init_grass_frac: float = 0.2


@dataclass(frozen=True)
class OmniD14Config(OmniConfig):
    """Stage 9c's control: the winter world's yearly average gut, all year."""
    plant_digest: float = 0.14


@dataclass(frozen=True)
class OmniWinterConfig(OmniConfig):
    """Stage 9c: plants are thin food half the time, so summer pays above the diet switch and winter below it."""
    plant_digest: float = 0.224
    winter_len: int = 60


@dataclass(frozen=True)
class OmniNeutralConfig(OmniConfig):
    """Stage 9a's control: half of each for every hunter, so the diet gene only drifts."""
    diet_fixed: bool = True


@dataclass(frozen=True)
class FatPreyRichConfig(FatPreyConfig):
    """Stage 5b v2: the same idea with the average meal back at about the control's 25, so hunting pays as well as it does there."""
    meal_share: float = 2.6


@dataclass(frozen=True)
class FatPreyFixedConfig(EvoConfig):
    """Stage 5b's control: hunters can see who is fat, but every meal is worth the same, so it should not matter."""
    fat_sense: bool = True


@dataclass(frozen=True)
class SignalConfig(PackConfig):
    """Stage 6b: the pack world, plus a signal that animals of the same species can see."""
    signals: bool = True


@dataclass(frozen=True)
class NichesConfig(TerrainConfig):
    """Stage 5a: size trades burrows against tough plants and safety; breeding is sexual.
    v2 (after pp-niche-001): no prey is immune, big prey are bigger meals, mates match in size.
    v3 (after pp-niche-002): size no longer touches predation; two fixed plant kinds, grass and
    shrubs, so prey can split by what they eat (big bodies digest shrubs, small ones live cheaply on grass).
    v4 (after pp-niche-003): small bodies digest grass best, so the middle size is second best at both.
    v5 (after pp-niche-004): grass had died out within minutes in v3 and v4; now both kinds keep their share."""
    sizes: bool = True
    mating: bool = True
    catch_scale: float = 0.1
    mate_tolerance: float = 0.2
    size_catch: bool = False
    plant_sigma: float = 0.0
    plant_kinds: bool = True
    size_tradeoff: bool = True
    kinds_persist: bool = True


class Species(NamedTuple):
    pos: jax.Array  # (N, 2) int32, (x, y)
    energy: jax.Array  # (N,) float32
    alive: jax.Array  # (N,) bool
    genes: jax.Array  # (N, 4) float32: speed, sight, breed, size
    gen: jax.Array  # (N,) int32: generations since the first founders
    tag: jax.Array  # (N,) int32: lineage marker, copied to children (evaluation only)


class State(NamedTuple):
    prey: Species
    pred: Species
    plant: jax.Array  # (size, size) bool
    plant_genes: jax.Array  # (size, size, 2) float32: spread, toughness
    t: jax.Array  # () int32
    terrain: jax.Array  # (size, size) int8: 0 open, WALL, BURROW; fixed for the episode
    bitten: jax.Array  # (size, size) bool: survived a bite and is regrowing
    prey_signal: jax.Array  # (max_prey,) bool: signalled on the last step (always False without signals)
    pred_signal: jax.Array  # (max_pred,) bool
    good: jax.Array  # () int32: the plant kind that feeds, 0 soft or 1 tough (fickle worlds; never observed)
    since_flip: jax.Array  # () int32: steps since the good kind last flipped


class GenePool(NamedTuple):
    """Genes of every living animal and plant across a batch of worlds, with
    sampling weights (1 = alive, 0 = empty)."""
    prey_genes: jax.Array
    prey_gen: jax.Array
    prey_w: jax.Array
    pred_genes: jax.Array
    pred_gen: jax.Array
    pred_w: jax.Array
    plant_genes: jax.Array
    plant_w: jax.Array
    prey_tag: jax.Array | None = None  # lineage marks, so a marked line survives a world's restart (Stage 11c)
    pred_tag: jax.Array | None = None


def sight_radius(g: jax.Array) -> jax.Array:
    return 1 + jnp.floor(3 * g + 0.5).astype(jnp.int32)


def is_night(cfg: EvoConfig, t) -> jax.Array:
    if not cfg.day_len:
        return jnp.bool_(False)
    return (t % cfg.day_len) >= cfg.day_len * (1 - cfg.night_frac)


def in_season(cfg: EvoConfig, t) -> jax.Array:
    """(size, size) bool: cells whose half of the map is in season (all of them without seasons)."""
    if not cfg.season_len:
        return jnp.ones((cfg.size, cfg.size), bool)
    left = jnp.arange(cfg.size)[None, :] < cfg.size // 2
    return jnp.broadcast_to(jnp.where((t // cfg.season_len) % 2 == 0, left, ~left), (cfg.size, cfg.size))


def mutate(key: jax.Array, genes: jax.Array, sigma: float) -> jax.Array:
    return jnp.clip(genes + sigma * jax.random.normal(key, genes.shape), 0.0, 1.0)


def _founders(key, n, mean, noise):
    return mutate(key, jnp.broadcast_to(jnp.asarray(mean, jnp.float32), (n, len(mean))), noise)


def _draw(key, n, genes, weights, fallback, gen=None):
    """Draw n genomes (and generations) from a weighted pool, copied without
    mutation; an empty pool gives the fallback genomes and generation 0."""
    total = weights.sum()
    p = jnp.where(total > 0, weights / jnp.maximum(total, 1e-9), 1.0 / weights.shape[0])
    idx = jax.random.choice(key, weights.shape[0], (n,), p=p)
    use = total > 0
    out = jnp.where(use, genes[idx], fallback)
    if gen is None:
        return out
    return out, jnp.where(use, gen[idx], 0)


def _plant_sigma(cfg: EvoConfig) -> float:
    return cfg.gene_sigma if cfg.plant_sigma < 0 else cfg.plant_sigma


def _plant_founders(cfg: EvoConfig, key, n):
    if not cfg.plant_kinds:
        return _founders(key, n, cfg.founder_plant, cfg.founder_noise)
    tough = jax.random.bernoulli(key, 0.5, (n,)).astype(jnp.float32)
    return jnp.stack([jnp.full((n,), cfg.founder_plant[0], jnp.float32), tough], -1)


def _terrain(cfg: EvoConfig, key: jax.Array) -> jax.Array:
    """Random straight wall segments (wrapping), then burrow cells on top."""
    t = jnp.zeros((cfg.size, cfg.size), jnp.int8)
    if cfg.walls:
        k_start, k_dir = jax.random.split(key)
        start = jax.random.randint(k_start, (cfg.walls, 2), 0, cfg.size)
        flat = jax.random.bernoulli(k_dir, 0.5, (cfg.walls,))
        step = jnp.where(flat[:, None], jnp.array([1, 0]), jnp.array([0, 1]))
        cells = (start[:, None, :] + jnp.arange(cfg.wall_len)[None, :, None] * step[:, None, :]) % cfg.size
        t = t.at[cells[..., 1], cells[..., 0]].set(WALL)
    if cfg.burrows:
        xy = jax.random.randint(jax.random.fold_in(key, 1), (cfg.burrows, 2), 0, cfg.size)
        t = t.at[xy[:, 1], xy[:, 0]].set(BURROW)
    return t


def _open_positions(key, n, terrain):
    """n random open cells (with replacement) as (x, y)."""
    size = terrain.shape[0]
    free = (terrain == 0).reshape(-1).astype(jnp.float32)
    idx = jax.random.choice(key, size * size, (n,), p=free / jnp.maximum(free.sum(), 1.0))
    return jnp.stack([idx % size, idx // size], -1).astype(jnp.int32)


def _spawn(cfg, key, n, n_alive, start_energy, pool_genes=None, pool_gen=None, pool_w=None, tag=0, terrain=None, pool_tag=None):
    k_pos, k_found, k_draw = jax.random.split(key, 3)
    if terrain is None:
        pos = jax.random.randint(k_pos, (n, 2), 0, cfg.size, dtype=jnp.int32)
    else:
        pos = _open_positions(k_pos, n, terrain)
    alive = jnp.arange(n) < n_alive
    genes = _founders(k_found, n, cfg.founder_animal, cfg.founder_noise)
    if cfg.brains and cfg.brain_spread:
        genes = genes.at[:, 4].set(jax.random.uniform(jax.random.fold_in(k_found, 4), (n,)))
    gen = jnp.zeros((n,), jnp.int32)
    tags = jnp.full((n,), tag, jnp.int32)
    if pool_genes is not None and pool_tag is not None:  # draw each founder's lineage mark along with its genes
        marked = jnp.concatenate([pool_genes, pool_tag[:, None].astype(pool_genes.dtype)], axis=1)
        drawn, gen = _draw(k_draw, n, marked, pool_w, jnp.concatenate([genes, jnp.zeros((n, 1), genes.dtype)], axis=1), pool_gen)
        genes, tags = drawn[:, :-1], jnp.round(drawn[:, -1]).astype(jnp.int32)
    elif pool_genes is not None:
        genes, gen = _draw(k_draw, n, pool_genes, pool_w, genes, pool_gen)
    energy = jnp.where(alive, start_energy, 0.0).astype(jnp.float32)
    return Species(pos, energy, alive, genes.astype(jnp.float32), jnp.where(alive, gen, 0), tags)


def reset(cfg: EvoConfig, key: jax.Array, pool: GenePool | None = None) -> State:
    k_prey, k_pred, k_grown, k_found, k_draw, k_terrain = jax.random.split(key, 6)
    cells = cfg.size * cfg.size
    terrain = _terrain(cfg, k_terrain)
    where = terrain if (cfg.walls or cfg.burrows) else None
    if pool is None:
        prey = _spawn(cfg, k_prey, cfg.max_prey, cfg.init_prey, cfg.prey_start_energy, terrain=where)
        pred = _spawn(cfg, k_pred, cfg.max_pred, cfg.init_pred, cfg.pred_start_energy, terrain=where)
    else:
        prey = _spawn(cfg, k_prey, cfg.max_prey, cfg.init_prey, cfg.prey_start_energy,
                      pool.prey_genes, pool.prey_gen, pool.prey_w, terrain=where, pool_tag=pool.prey_tag)
        pred = _spawn(cfg, k_pred, cfg.max_pred, cfg.init_pred, cfg.pred_start_energy,
                      pool.pred_genes, pool.pred_gen, pool.pred_w, terrain=where, pool_tag=pool.pred_tag)
    plant = (jax.random.uniform(k_grown, (cfg.size, cfg.size)) < cfg.init_grass_frac) & (terrain == 0)
    plant_genes = _plant_founders(cfg, k_found, cells)
    if pool is not None:
        plant_genes = _draw(k_draw, cells, pool.plant_genes, pool.plant_w, plant_genes)
    plant_genes = plant_genes.reshape(cfg.size, cfg.size, PLANT_GENES)
    good = jax.random.bernoulli(jax.random.fold_in(key, 98)).astype(jnp.int32) if cfg.fickle else jnp.int32(0)
    return State(prey, pred, plant, jnp.where(plant[..., None], plant_genes, 0.0), jnp.int32(0), terrain,
                 jnp.zeros((cfg.size, cfg.size), bool), jnp.zeros(cfg.max_prey, bool), jnp.zeros(cfg.max_pred, bool),
                 good, jnp.int32(0))


def gene_pool(states: State) -> GenePool:
    """Pool the living animals and plants of a batch of worlds (leading axis E)."""
    def animals(sp: Species):
        return (sp.genes.reshape(-1, ANIMAL_GENES), sp.gen.reshape(-1),
                sp.alive.reshape(-1).astype(jnp.float32))

    prey_genes, prey_gen, prey_w = animals(states.prey)
    pred_genes, pred_gen, pred_w = animals(states.pred)
    return GenePool(prey_genes, prey_gen, prey_w, pred_genes, pred_gen, pred_w,
                    states.plant_genes.reshape(-1, PLANT_GENES),
                    states.plant.reshape(-1).astype(jnp.float32),
                    states.prey.tag.reshape(-1), states.pred.tag.reshape(-1))


def _species_from_list(agents, n) -> Species:
    pos = np.zeros((n, 2), np.int32)
    energy = np.zeros(n, np.float32)
    alive = np.zeros(n, bool)
    genes = np.zeros((n, ANIMAL_GENES), np.float32)
    for i, (x, y, e, *g) in enumerate(agents):
        pos[i] = (x, y)
        energy[i] = e
        alive[i] = True
        genes[i] = tuple(g) + (FOUNDER_SIZE,) * (ANIMAL_GENES - len(g))
    return Species(jnp.asarray(pos), jnp.asarray(energy), jnp.asarray(alive), jnp.asarray(genes),
                   jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32))


def make_state(cfg: EvoConfig, prey=(), pred=(), plants=(), t=0, terrain=None) -> State:
    """Hand-placed state for tests. Animals are (x, y, energy, speed, sight, breed[, size]);
    plants are (x, y, spread, toughness); terrain is a (size, size) map of 0, WALL, BURROW."""
    plant = np.zeros((cfg.size, cfg.size), bool)
    plant_genes = np.zeros((cfg.size, cfg.size, PLANT_GENES), np.float32)
    for x, y, spread, tough in plants:
        plant[y, x] = True
        plant_genes[y, x] = (spread, tough)
    return State(_species_from_list(prey, cfg.max_prey), _species_from_list(pred, cfg.max_pred),
                 jnp.asarray(plant), jnp.asarray(plant_genes), jnp.int32(t),
                 jnp.zeros((cfg.size, cfg.size), jnp.int8) if terrain is None else jnp.asarray(terrain, jnp.int8),
                 jnp.zeros((cfg.size, cfg.size), bool), jnp.zeros(cfg.max_prey, bool), jnp.zeros(cfg.max_pred, bool),
                 jnp.int32(0), jnp.int32(0))


HILL_WAVES = 10


def height_map(cfg: EvoConfig) -> np.ndarray:
    """Stage 8a: the height of every cell, in [0, 1], indexed [y, x]. A fixed sum of waves whose
    periods divide the map, so the torus stays seamless; hill_seed picks the map."""
    rng = np.random.default_rng(cfg.hill_seed)
    n = cfg.size
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = np.zeros((n, n))
    for _ in range(HILL_WAVES):
        kx, ky = 0, 0
        while max(abs(kx), abs(ky)) < 2:  # no wave longer than half the map
            kx, ky = rng.integers(-6, 7, 2)
        h += np.sin(2 * np.pi * (kx * x + ky * y) / n + rng.uniform(0, 2 * np.pi)) / max(abs(kx), abs(ky))
    return ((h - h.min()) / (h.max() - h.min())).astype(np.float32)


def thicket_map(cfg: EvoConfig) -> np.ndarray:
    """Stage 8b: which cells are thickets, indexed [y, x]: the top `thickets` share of a fixed sum of waves,
    so they form seamless patches. thicket_seed picks the map."""
    rng = np.random.default_rng(10_000 + cfg.thicket_seed)
    n = cfg.size
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = np.zeros((n, n))
    for _ in range(HILL_WAVES):
        kx, ky = 0, 0
        while max(abs(kx), abs(ky)) < 2:
            kx, ky = rng.integers(-6, 7, 2)
        h += np.sin(2 * np.pi * (kx * x + ky * y) / n + rng.uniform(0, 2 * np.pi)) / max(abs(kx), abs(ky))
    return h >= np.quantile(h, 1 - cfg.thickets)


def _walk(cfg: EvoConfig, before: Species, after: Species) -> Species:
    """Stage 8b: an animal pays move_cost for every cell it moved (standing still is free)."""
    d = jnp.abs(after.pos - before.pos)
    cells = jnp.minimum(d, cfg.size - d).sum(-1)
    return after._replace(energy=after.energy - cfg.move_cost * cells * after.alive)


def sight_lines(view: int):
    """For every cell of the view (indexed [dy, dx] from the top left), the cells strictly between
    it and the centre on a straight line, and how far along the line each one is.
    Returns (iy, ix, frac, valid), each (view, view, view // 2 - 1)."""
    r = view // 2
    k = max(r - 1, 1)
    iy, ix = np.full((view, view, k), r), np.full((view, view, k), r)
    frac, valid = np.zeros((view, view, k), np.float32), np.zeros((view, view, k), bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = max(abs(dx), abs(dy))
            for j in range(1, d):
                t = j / d
                iy[dy + r, dx + r, j - 1] = r + int(np.rint(dy * t))
                ix[dy + r, dx + r, j - 1] = r + int(np.rint(dx * t))
                frac[dy + r, dx + r, j - 1], valid[dy + r, dx + r, j - 1] = t, True
    return iy, ix, frac, valid


def _visible(cfg: EvoConfig, heights, h_self):
    """(N, view, view) bool: no cell between the animal and a cell rises above the line from its
    eye to that cell's ground."""
    iy, ix, frac, valid = sight_lines(cfg.view)
    between = heights[:, iy, ix]  # (N, view, view, K)
    eye = (h_self + cfg.eye_height)[:, None, None, None]
    line = eye + (heights[..., None] - eye) * frac
    return ~(jnp.asarray(valid) & (between > line + 1e-6)).any(-1)


def _climb(cfg: EvoConfig, before: Species, after: Species) -> Species:
    """Stage 8a: an animal pays climb_cost for every unit of height it gains."""
    h = jnp.asarray(height_map(cfg))
    up = jnp.maximum(h[after.pos[:, 1], after.pos[:, 0]] - h[before.pos[:, 1], before.pos[:, 0]], 0.0)
    return after._replace(energy=after.energy - cfg.climb_cost * up * after.alive)


def observe(cfg: EvoConfig, state: State) -> tuple[jax.Array, jax.Array]:
    maps = jnp.stack(
        [
            (state.plant & ~state.bitten).astype(jnp.float32),  # edible plants
            state.plant_genes[..., 1] * state.plant,
            pp._count_map(state.prey, cfg.size),
            pp._count_map(state.pred, cfg.size),
        ],
        axis=-1,
    )
    if cfg.signals:  # each species sees only its own signals
        heard = {"prey": pp._count_map(state.prey._replace(alive=state.prey.alive & state.prey_signal), cfg.size),
                 "pred": pp._count_map(state.pred._replace(alive=state.pred.alive & state.pred_signal), cfg.size)}
    terrain = jnp.stack([state.terrain == WALL, state.terrain == BURROW], -1).astype(jnp.float32)
    r = cfg.view // 2
    offs = jnp.arange(-r, r + 1)
    cheb = jnp.maximum(jnp.abs(offs)[:, None], jnp.abs(offs)[None, :])  # (view, view)
    hmap = jnp.asarray(height_map(cfg)) if cfg.hills else None
    cover = jnp.asarray(thicket_map(cfg))[..., None].astype(jnp.float32) if cfg.thickets > 0 else None
    fat = (jnp.zeros((cfg.size, cfg.size), jnp.float32).at[state.prey.pos[:, 1], state.prey.pos[:, 0]]
           .add(state.prey.energy * state.prey.alive / cfg.prey_energy_cap))[..., None] if cfg.fat_sense else None

    def one(sp: Species, cap: float, kind: str) -> jax.Array:
        w = pp._windows(maps, sp.pos, cfg.view, cfg.size)
        radius = jnp.where(is_night(cfg, state.t), 1, sight_radius(sp.genes[:, 1]))
        if cfg.hills:  # high ground sees further by day; hills hide what is behind them
            heights = pp._windows(hmap[..., None], sp.pos, cfg.view, cfg.size)[..., 0]  # (N, view, view)
            h_self = hmap[sp.pos[:, 1], sp.pos[:, 0]]
            bonus = jnp.floor(cfg.hill_sight * 2 * jnp.maximum(h_self - 0.5, 0.0) + 0.5).astype(jnp.int32)
            radius = jnp.where(is_night(cfg, state.t), radius, radius + bonus)
            seen = (cheb[None] <= radius[:, None, None]) & _visible(cfg, heights, h_self)
        else:
            seen = cheb[None] <= radius[:, None, None]
        if cfg.thickets > 0:  # an animal in a thicket shows only to the next cell
            in_cover = pp._windows(cover, sp.pos, cfg.view, cfg.size)[..., 0] > 0
            hidden = in_cover & (cheb[None] > 1)
            if cfg.cover_hides_prey:
                w = w.at[..., 2].set(jnp.where(hidden, 0.0, w[..., 2]))
            if cfg.cover_hides_pred:
                w = w.at[..., 3].set(jnp.where(hidden, 0.0, w[..., 3]))
        w = jnp.where(seen[..., None], w, 0.0)
        if cfg.stealth_radius > 0 and kind == "prey":  # hunters are camouflaged beyond this range
            near = jnp.abs(jnp.arange(cfg.view) - cfg.view // 2) <= cfg.stealth_radius
            w = w.at[..., 3].set(w[..., 3] * (near[:, None] & near[None, :]))
        own = jnp.concatenate([(sp.energy / cap)[:, None], sp.genes[:, :3]], axis=-1)  # (N, 4)
        own = jnp.broadcast_to(own[:, None, None, :], w.shape[:-1] + (4,))
        size = jnp.broadcast_to(sp.genes[:, None, None, 3:4], w.shape[:-1] + (1,))
        ground = jnp.where(seen[..., None], pp._windows(terrain, sp.pos, cfg.view, cfg.size), 0.0)
        # New channels go last, so older brains load unchanged (their inputs are zero-padded).
        # Signals are calls: heard across the whole view, by day or night, unlike what is seen.
        parts = [w, own, ground, size] + ([pp._windows(heard[kind][..., None], sp.pos, cfg.view, cfg.size)]
                                          if cfg.signals else [])
        if cfg.hills:  # the height of each visible cell relative to the animal's own
            parts.append(jnp.where(seen, heights - h_self[:, None, None], 0.0)[..., None])
        if cfg.thickets > 0:  # where the thickets are, within sight
            parts.append(jnp.where(seen, in_cover, False).astype(jnp.float32)[..., None])
        if cfg.fat_sense:  # how much energy the prey in each visible cell hold
            parts.append(jnp.where(seen[..., None], pp._windows(fat, sp.pos, cfg.view, cfg.size), 0.0))
        if cfg.omnivore:  # the animal's own diet gene
            parts.append(jnp.broadcast_to(sp.genes[:, None, None, 5:6], w.shape[:-1] + (1,)))
        if cfg.brains:  # last, so the network can read it to size the brain
            parts.append(jnp.broadcast_to(sp.genes[:, None, None, 4:5], w.shape[:-1] + (1,)))
        obs = jnp.concatenate(parts, axis=-1)
        return jnp.where(sp.alive[:, None, None, None], obs, 0.0).astype(jnp.bfloat16)

    return one(state.prey, cfg.prey_energy_cap, "prey"), one(state.pred, cfg.pred_energy_cap, "pred")


def obs_channels(cfg: EvoConfig) -> int:
    return OBS_CHANNELS + int(cfg.signals) + int(cfg.hills) + int(cfg.thickets > 0) + int(cfg.fat_sense) + int(cfg.omnivore) + int(cfg.brains)


def num_actions(cfg: EvoConfig) -> int:
    return NUM_ACTIONS * (2 if cfg.signals else 1)


GENE_NAMES = {"animal": ("speed", "sight", "breed", "size", "brain", "diet"), "plant": ("spread", "tough")}


class StepInfo(NamedTuple):
    catches: jax.Array
    prey_births: jax.Array
    pred_births: jax.Array
    prey_deaths: jax.Array
    pred_deaths: jax.Array
    prey_alive: jax.Array
    pred_alive: jax.Array
    prey_pred_dist_sum: jax.Array
    prey_pred_dist_count: jax.Array
    pred_pred_dist_sum: jax.Array
    pred_pred_dist_count: jax.Array
    prey_gene_sum: jax.Array  # (4,)
    prey_gene_sq: jax.Array
    pred_gene_sum: jax.Array
    pred_gene_sq: jax.Array
    prey_gen_sum: jax.Array
    pred_gen_sum: jax.Array
    plant_gene_sum: jax.Array  # (2,)
    plant_gene_sq: jax.Array
    plant_count: jax.Array
    plant_cover: jax.Array  # fraction of cells with a plant
    prey_near: jax.Array  # living prey with a predator within THREAT cells
    prey_near_burrow: jax.Array  # ... of which sit in a burrow
    prey_far: jax.Array
    prey_far_burrow: jax.Array
    good_bites: jax.Array  # fickle worlds: bites of the kind that feeds, and of the kind that poisons
    bad_bites: jax.Array
    eaten_energy: jax.Array  # energy held by the prey eaten this step
    eaten_baseline: jax.Array  # the same if each had held the living prey's mean energy
    pred_bites: jax.Array  # omnivore worlds: plants eaten by hunters
    pred_plant_gain: jax.Array  # energy hunters got from plants
    pred_meat_gain: jax.Array  # energy hunters got from catches


THREAT = 3  # Chebyshev cells (wrapped) at which a predator counts as close


def _burrow_use(cfg: EvoConfig, prey: Species, pred: Species, terrain) -> tuple:
    d = jnp.abs(prey.pos[:, None, :] - pred.pos[None, :, :])
    d = jnp.minimum(d, cfg.size - d).max(-1)  # (prey, pred) wrapped Chebyshev distance
    near = prey.alive & (jnp.where(pred.alive[None, :], d, cfg.size) <= THREAT).any(-1)
    far = prey.alive & ~near
    hidden = terrain[prey.pos[:, 1], prey.pos[:, 0]] == BURROW
    count = lambda m: m.sum().astype(jnp.int32)
    return count(near), count(near & hidden), count(far), count(far & hidden)


def _move(cfg: EvoConfig, key, sp: Species, act, blocked, big_blocked=None) -> Species:
    """blocked: (size, size) bool, cells this species cannot enter. big_blocked: cells only
    animals smaller than burrow_size can enter (used when sizes are on)."""
    def go(pos):
        nxt = pp.move(pos, act, cfg.size)
        stop = blocked[nxt[:, 1], nxt[:, 0]]
        if big_blocked is not None:
            stop = stop | (big_blocked[nxt[:, 1], nxt[:, 0]] & (sp.genes[:, 3] >= cfg.burrow_size))
        return jnp.where(stop[:, None], pos, nxt)

    once = go(sp.pos)
    twice = go(once)
    fast = jax.random.uniform(key, sp.alive.shape) < sp.genes[:, 0]
    pos = jnp.where(fast[:, None], twice, once)
    return sp._replace(pos=jnp.where(sp.alive[:, None], pos, sp.pos))


def _kept(cfg: EvoConfig, sp: Species):
    """Share of a meal an animal keeps: all of it, or 1 - brain_tax * brain gene (Stage 7b)."""
    if cfg.brains and cfg.brain_tax > 0:
        return 1.0 - cfg.brain_tax * sp.genes[:, 4]
    return 1.0


def _meat_share(cfg: EvoConfig, pred: Species):
    """Share of a catch a hunter keeps for its diet gene: all of it, outside omnivore worlds (Stage 9a)."""
    if not cfg.omnivore:
        return 1.0
    return 0.5 if cfg.diet_fixed else pred.genes[:, 5]


def _plant_value(cfg: EvoConfig, t):
    """What a plant feeds at step t: 1, or winter_value in winter (Stage 9c)."""
    if not cfg.winter_len:
        return 1.0
    return jnp.where((t // cfg.winter_len) % 2 == 0, 1.0, cfg.winter_value)


def _metabolise(cfg: EvoConfig, sp: Species, cap: float, base: float = 1.0, speed_cost: float | None = None) -> Species:
    radius = sight_radius(sp.genes[:, 1])
    cost = base + (cfg.speed_cost if speed_cost is None else speed_cost) * sp.genes[:, 0] ** 2 + cfg.sight_cost * (radius - 1) / 3
    if cfg.sizes:
        cost = cost + cfg.size_cost * sp.genes[:, 3]
    if cfg.brains:
        cost = cost + cfg.brain_cost * sp.genes[:, 4]
    energy = jnp.minimum(sp.energy - cost * sp.alive, cap)
    return sp._replace(energy=jnp.where(sp.alive, energy, 0.0))


def catch_chance(cfg: EvoConfig, pred_size, prey_size):
    """Chance that a predator of pred_size catches a prey of prey_size on its cell (sizes on)."""
    if cfg.catch_scale > 0:
        return jax.nn.sigmoid((jnp.asarray(pred_size) - jnp.asarray(prey_size)) / cfg.catch_scale)
    return (jnp.asarray(prey_size) <= jnp.asarray(pred_size) + cfg.catch_margin).astype(jnp.float32)


def _predation(cfg: EvoConfig, key, prey: Species, pred: Species) -> tuple[jax.Array, jax.Array, jax.Array]:
    """predprey.predation, except that with sizes on each predator attacks the lowest-index prey on
    its cell and succeeds with catch_chance. Returns (caught, eaten, energy gained per predator):
    with sizes on, a meal is catch_energy * (0.5 + prey size)."""
    if cfg.pack_size > 1:
        return _pack_predation(cfg, prey, pred)
    if not (cfg.sizes and cfg.size_catch):
        caught, eaten = pp.predation(prey, pred)
        if cfg.stored_meals:  # the same choice pp.predation made: the lowest-index prey on the predator's cell
            same = jnp.all(pred.pos[:, None, :] == prey.pos[None, :, :], axis=-1) & pred.alive[:, None] & prey.alive[None, :]
            return caught, eaten, cfg.meal_share * prey.energy[jnp.argmax(same, axis=1)] * caught
        return caught, eaten, cfg.catch_energy * caught
    same = (
        jnp.all(pred.pos[:, None, :] == prey.pos[None, :, :], axis=-1)
        & pred.alive[:, None]
        & prey.alive[None, :]
    )
    if cfg.catch_scale <= 0:  # the hard cut-off: too-big prey are not even attacked
        same = same & (catch_chance(cfg, pred.genes[:, None, 3], prey.genes[None, :, 3]) > 0)
    has = same.any(axis=1)
    choice = jnp.argmax(same, axis=1)
    prey_size = prey.genes[choice, 3]
    if cfg.catch_scale > 0:
        has = has & (jax.random.uniform(key, has.shape) < catch_chance(cfg, pred.genes[:, 3], prey_size))
    caught = pp._first_claim(has, choice)
    eaten = jnp.zeros(prey.alive.shape[0], jnp.int32).at[choice].max(caught.astype(jnp.int32)) > 0
    return caught, eaten, cfg.catch_energy * (0.5 + prey_size) * caught


def _pack_predation(cfg: EvoConfig, prey: Species, pred: Species):
    """Each predator hunts the lowest-index prey within one cell (wrapped Chebyshev); a prey with at
    least pack_size hunters is eaten, and they share a meal of pack_size * catch_energy. With flank on,
    the hunters must also include two exactly opposite each other through the prey (west and east,
    north and south, or diagonal corners), so two hunters on the same side never count."""
    d = jnp.abs(pred.pos[:, None, :] - prey.pos[None, :, :])
    d = jnp.minimum(d, cfg.size - d).max(-1)
    near = (d <= 1) & pred.alive[:, None] & prey.alive[None, :]
    has = near.any(axis=1)
    choice = jnp.argmax(near, axis=1)
    hunters = jnp.zeros(prey.alive.shape[0], jnp.int32).at[choice].add(has.astype(jnp.int32))
    eaten = prey.alive & (hunters >= cfg.pack_size)
    if cfg.flank:
        off = pred.pos - prey.pos[choice]
        off = (off + cfg.size // 2) % cfg.size - cfg.size // 2  # wrapped, in [-size/2, size/2)
        cell = jnp.where(has, (off[:, 0] + 1) * 3 + (off[:, 1] + 1), 4)  # 0..8 around the prey, 4 = its own cell
        occ = jnp.zeros((prey.alive.shape[0], 9), jnp.int32).at[choice, cell].max(has.astype(jnp.int32)) > 0
        eaten = eaten & (occ[:, :4] & occ[:, 8:4:-1]).any(-1)  # cell c and 8 - c are exact opposites
    caught = has & eaten[choice]
    meal = cfg.pack_size * cfg.catch_energy / jnp.maximum(hunters[choice], 1)
    return caught, eaten, jnp.where(caught, meal, 0.0)


def _mates(cfg: EvoConfig, sp: Species) -> tuple[jax.Array, jax.Array]:
    """For each animal, whether another living animal of its species (within mate_tolerance in
    size) is within mate_radius, and the index of the nearest one (lowest index on ties)."""
    n = sp.alive.shape[0]
    d = jnp.abs(sp.pos[:, None, :] - sp.pos[None, :, :])
    d = jnp.minimum(d, cfg.size - d).max(-1)
    ok = sp.alive[None, :] & ~jnp.eye(n, dtype=bool)
    if cfg.mate_tolerance < 1:
        ok = ok & (jnp.abs(sp.genes[:, None, 3] - sp.genes[None, :, 3]) <= cfg.mate_tolerance)
    d = jnp.where(ok, d, cfg.size * 2)
    mate = jnp.argmin(d, axis=1)
    return d[jnp.arange(n), mate] <= cfg.mate_radius, mate


def _reproduce(cfg: EvoConfig, key, sp: Species, repro_energy: float) -> tuple[Species, jax.Array]:
    """Like predprey.reproduce, but each animal's threshold comes from its breed
    gene and the child inherits mutated genes, generation + 1 and the tag. With
    mating on, a ready animal also needs a mate nearby, and the child takes each
    gene from one parent or the other."""
    n = sp.alive.shape[0]
    parents = sp.alive & (sp.energy >= repro_energy * (0.5 + cfg.breed_span * sp.genes[:, 2]))
    inherited = sp.genes
    if cfg.mating:
        key, k_cross = jax.random.split(key)
        has_mate, mate = _mates(cfg, sp)
        parents = parents & has_mate
        inherited = jnp.where(jax.random.bernoulli(k_cross, 0.5, sp.genes.shape), sp.genes, sp.genes[mate])
    free = ~sp.alive
    rank = jnp.cumsum(parents) - 1
    can = parents & (rank < free.sum())
    free_idx = jnp.nonzero(free, size=n, fill_value=0)[0]
    target = jnp.where(can, free_idx[jnp.clip(rank, 0, n - 1)], n)  # n = dropped
    half = (sp.energy - cfg.birth_cost) / 2
    child_genes = mutate(key, inherited, cfg.gene_sigma)
    return Species(
        pos=sp.pos.at[target].set(sp.pos, mode="drop"),
        energy=jnp.where(can, half, sp.energy).at[target].set(half, mode="drop"),
        alive=sp.alive.at[target].set(True, mode="drop"),
        genes=sp.genes.at[target].set(child_genes, mode="drop"),
        gen=sp.gen.at[target].set(sp.gen + 1, mode="drop"),
        tag=sp.tag.at[target].set(sp.tag, mode="drop"),
    ), can


def _graze(cfg: EvoConfig, key, prey: Species, plant, plant_genes, bitten, good=0, factor=1.0):
    """Prey on an unbitten plant eat it (lowest index first). A plant survives with
    probability equal to its toughness, but is then bitten until it regrows; the
    grazer gains less from tough plants."""
    xs, ys = prey.pos[:, 0], prey.pos[:, 1]
    on = prey.alive & plant[ys, xs] & ~bitten[ys, xs]
    eats = pp._first_claim(on, ys * cfg.size + xs)
    tough = plant_genes[ys, xs, 1]
    if cfg.sizes:
        size = prey.genes[:, 3]
        hard = tough * (1 - size)  # big grazers digest tough plants fully
        if cfg.size_tradeoff:
            hard = hard + (1 - tough) * size  # and small ones soft plants
        tough = hard
    gain = cfg.grass_energy * (1 - 0.5 * tough) * eats * _kept(cfg, prey)
    tasty = (plant_genes[ys, xs, 1] >= 0.5).astype(jnp.int32) == good
    if cfg.fickle:  # the kinds feed or poison in full; toughness does not discount them
        gain = jnp.where(tasty, cfg.grass_energy * _kept(cfg, prey), -cfg.poison) * eats
    removed = eats & (jax.random.uniform(key, eats.shape) >= plant_genes[ys, xs, 1]) & (not cfg.kinds_persist)
    kill = jnp.zeros(plant.shape, jnp.int32).at[ys, xs].max(removed.astype(jnp.int32)) > 0
    bit = jnp.zeros(plant.shape, jnp.int32).at[ys, xs].max(eats.astype(jnp.int32)) > 0
    plant = plant & ~kill
    good_bites = (eats & tasty).sum().astype(jnp.int32)
    bad_bites = (eats & ~tasty).sum().astype(jnp.int32)
    return prey._replace(energy=prey.energy + gain * factor), plant, (bitten | bit) & plant, good_bites, bad_bites


def _plants(cfg: EvoConfig, key, plant, genes, ground, bitten, season):
    """Natural death, then seeding into a bare 4-neighbour (a cell seeded from
    several neighbours takes the highest random score), then germination."""
    k_die, k_dir, k_seed, k_mut, k_prio, k_germ, k_found, k_src, k_gmut, k_regrow = jax.random.split(key, 10)
    shape = plant.shape
    spread, tough = genes[..., 0], genes[..., 1]
    p_die = cfg.die_rate * (cfg.plant_base_death + (1 - cfg.plant_base_death) * spread)
    p_die = jnp.where(season, p_die, p_die * cfg.offseason_death)
    plant = plant & ~(jax.random.uniform(k_die, shape) < p_die)

    bare = ~plant & ground  # plants never take walls or burrows
    # nb_bare[y, x, i]: the neighbour of (x, y) in direction i is bare.
    nb_bare = jnp.stack([jnp.roll(bare, (-dy, -dx), (0, 1)) for dx, dy in NEIGHBOURS], -1)
    direction = jnp.argmax(jnp.where(nb_bare, jax.random.uniform(k_dir, nb_bare.shape), -1.0), -1)
    p_seed = cfg.seed_rate * spread * (1 - cfg.tough_seed_cost * tough * (not cfg.kinds_persist))
    seeds = plant & season & ~bitten & nb_bare.any(-1) & (jax.random.uniform(k_seed, shape) < p_seed)
    child = mutate(k_mut, genes, _plant_sigma(cfg))
    # incoming[y, x, i]: the source at (x - dx, y - dy) seeds into (x, y).
    incoming = jnp.stack([jnp.roll(seeds & (direction == i), (dy, dx), (0, 1))
                          for i, (dx, dy) in enumerate(NEIGHBOURS)], -1)
    child_in = jnp.stack([jnp.roll(child, (dy, dx), (0, 1)) for dx, dy in NEIGHBOURS], -2)
    pick = jnp.argmax(jnp.where(incoming, jax.random.uniform(k_prio, incoming.shape), -1.0), -1)
    picked = jnp.take_along_axis(child_in, pick[..., None, None], axis=-2)[..., 0, :]
    gets = incoming.any(-1)
    genes = jnp.where(gets[..., None], picked, genes)
    plant = plant | gets

    cells = shape[0] * shape[1]
    germ = ~plant & ground & season & (jax.random.uniform(k_germ, shape) < cfg.germinate_rate)
    founders = _plant_founders(cfg, k_found, cells)
    weights = plant.reshape(-1).astype(jnp.float32)
    src = _draw(k_src, cells, genes.reshape(-1, PLANT_GENES), weights, founders)
    src = jnp.where(weights.sum() > 0, mutate(k_gmut, src, _plant_sigma(cfg)), src)
    if cfg.kinds_persist:
        src = founders
    genes = jnp.where(germ[..., None], src.reshape(genes.shape), genes)
    plant = plant | germ
    bitten = bitten & plant & ~(jax.random.uniform(k_regrow, shape) < cfg.bite_regrow)
    return plant, jnp.where(plant[..., None], genes, 0.0), bitten


def advance(cfg: EvoConfig, key: jax.Array, state: State, prey_act, pred_act) -> pp.StepResult:
    k_prey, k_pred, k_graze, k_rprey, k_rpred, k_plant = jax.random.split(key, 6)
    prey, pred = state.prey, state.pred
    prey_was, pred_was = prey.alive, pred.alive
    prey_sig = (prey_act >= NUM_ACTIONS) & prey_was if cfg.signals else jnp.zeros_like(prey_was)
    pred_sig = (pred_act >= NUM_ACTIONS) & pred_was if cfg.signals else jnp.zeros_like(pred_was)
    prey_act, pred_act = prey_act % NUM_ACTIONS, pred_act % NUM_ACTIONS

    # 1. Movement; fast animals sometimes take two cells.
    terrain = state.terrain
    if cfg.sizes:
        prey = _move(cfg, k_prey, prey, prey_act, terrain == WALL, terrain == BURROW)
    else:
        prey = _move(cfg, k_prey, prey, prey_act, terrain == WALL)
    pred = _move(cfg, k_pred, pred, pred_act, terrain != 0)
    if cfg.hills:
        prey, pred = _climb(cfg, state.prey, prey), _climb(cfg, state.pred, pred)
    if cfg.move_cost > 0:
        prey, pred = _walk(cfg, state.prey, prey), _walk(cfg, state.pred, pred)

    # 2. Predation.
    caught, eaten, meal = _predation(cfg, jax.random.fold_in(k_pred, 7), prey, pred)
    meat_gain = meal * _kept(cfg, pred) * _meat_share(cfg, pred)
    pred = pred._replace(energy=pred.energy + meat_gain)
    eaten_energy = (prey.energy * eaten).sum()  # how fat the eaten were, against how fat the living were
    living = (prey.energy * prey.alive).sum() / jnp.maximum(prey.alive.sum(), 1)
    eaten_baseline = eaten.sum() * living
    prey = prey._replace(alive=prey.alive & ~eaten)

    # 3. Grazing.
    plant_value = _plant_value(cfg, state.t)
    prey, plant, bitten, good_bites, bad_bites = _graze(cfg, k_graze, prey, state.plant, state.plant_genes,
                                                        state.bitten, state.good, plant_value)

    pred_bites, pred_plant_gain = jnp.int32(0), jnp.float32(0.0)
    if cfg.omnivore:  # hunters take what the grazers left this step
        before = pred.energy
        pred, plant, bitten, g, b = _graze(cfg, jax.random.fold_in(k_graze, 1), pred, plant, state.plant_genes, bitten, state.good,
                                           (1.0 - _meat_share(cfg, pred)) * cfg.plant_digest * plant_value)
        pred_bites, pred_plant_gain = g + b, (pred.energy - before).sum()

    # 4. Metabolism (with trait and signal costs) and caps.
    prey = prey._replace(energy=prey.energy - cfg.signal_cost * prey_sig)
    pred = pred._replace(energy=pred.energy - cfg.signal_cost * pred_sig)
    prey = _metabolise(cfg, prey, cfg.prey_energy_cap)
    pred = _metabolise(cfg, pred, cfg.pred_energy_cap, cfg.pred_base_cost, cfg.pred_speed_cost if cfg.pred_speed_cost >= 0 else None)

    # 5. Starvation. Deaths are recorded before any slot is reused.
    prey = prey._replace(alive=prey.alive & (prey.energy > 0))
    pred = pred._replace(alive=pred.alive & (pred.energy > 0))
    prey_died = prey_was & ~prey.alive
    pred_died = pred_was & ~pred.alive

    # 6. Reproduction with inherited, mutated genes.
    prey, prey_born = _reproduce(cfg, k_rprey, prey, cfg.prey_repro_energy)
    pred, pred_born = _reproduce(cfg, k_rpred, pred, cfg.pred_repro_energy)

    # 7. Plants die, seed and germinate.
    plant, plant_genes, bitten = _plants(cfg, k_plant, plant, state.plant_genes, terrain == 0, bitten,
                                         in_season(cfg, state.t))

    t = state.t + 1
    done = (t >= cfg.max_steps) | ~prey.alive.any() | ~pred.alive.any()
    good, since_flip = state.good, state.since_flip + 1
    if cfg.fickle:
        flip = jax.random.uniform(jax.random.fold_in(key, 99)) < cfg.flip_rate
        good = jnp.where(flip, 1 - good, good)
        since_flip = jnp.where(flip, 0, since_flip)

    def reward(was, died, born):
        r = jnp.where(died, cfg.death_reward, cfg.alive_reward) + cfg.birth_reward * born
        return jnp.where(was, r, 0.0).astype(jnp.float32)

    def gene_stats(sp: Species):
        w = sp.alive[:, None].astype(jnp.float32)
        return (sp.genes * w).sum(0), (sp.genes ** 2 * w).sum(0), (sp.gen * sp.alive).sum().astype(jnp.float32)

    prey_sum, prey_sq, prey_gen = gene_stats(prey)
    pred_sum, pred_sq, pred_gen = gene_stats(pred)
    pw = plant[..., None].astype(jnp.float32)
    near, near_hidden, far, far_hidden = _burrow_use(cfg, prey, pred, terrain)
    pp_sum, pp_count = pp.nearest_dist(prey, pred, cfg.size)
    qq_sum, qq_count = pp.nearest_dist(pred, pred, cfg.size, exclude_self=True)
    info = StepInfo(
        catches=eaten.sum().astype(jnp.int32),  # prey eaten (equals predators that ate unless hunting in packs)
        prey_births=prey_born.sum().astype(jnp.int32),
        pred_births=pred_born.sum().astype(jnp.int32),
        prey_deaths=prey_died.sum().astype(jnp.int32),
        pred_deaths=pred_died.sum().astype(jnp.int32),
        prey_alive=prey.alive.sum().astype(jnp.int32),
        pred_alive=pred.alive.sum().astype(jnp.int32),
        prey_pred_dist_sum=pp_sum,
        prey_pred_dist_count=pp_count,
        pred_pred_dist_sum=qq_sum,
        pred_pred_dist_count=qq_count,
        prey_gene_sum=prey_sum, prey_gene_sq=prey_sq,
        pred_gene_sum=pred_sum, pred_gene_sq=pred_sq,
        prey_gen_sum=prey_gen, pred_gen_sum=pred_gen,
        plant_gene_sum=(plant_genes * pw).sum((0, 1)),
        plant_gene_sq=(plant_genes ** 2 * pw).sum((0, 1)),
        plant_count=plant.sum().astype(jnp.int32),
        plant_cover=plant.mean(),
        prey_near=near, prey_near_burrow=near_hidden, prey_far=far, prey_far_burrow=far_hidden,
        good_bites=good_bites, bad_bites=bad_bites, eaten_energy=eaten_energy, eaten_baseline=eaten_baseline,
        pred_bites=pred_bites, pred_plant_gain=pred_plant_gain, pred_meat_gain=jnp.sum(meat_gain),
    )
    return pp.StepResult(
        # A slot that died this step is silent, even if a newborn took it.
        state=State(prey, pred, plant, plant_genes, t, terrain, bitten, prey_sig & ~prey_died, pred_sig & ~pred_died,
                    good, since_flip),
        prey_reward=reward(prey_was, prey_died, prey_born),
        pred_reward=reward(pred_was, pred_died, pred_born),
        prey_done=prey_died | ~prey_was | done,
        pred_done=pred_died | ~pred_was | done,
        done=done,
        info=info,
    )


def batch_step(cfg: EvoConfig, key: jax.Array, states: State, prey_act, pred_act):
    """Step a batch of worlds. Worlds that ended are reset with founders drawn
    from the living animals and plants of every world, so lineages carry on.
    `info` still describes the finished worlds. Returns (result, prey_obs, pred_obs)."""
    n = prey_act.shape[0]
    k_step, k_reset = jax.random.split(key)
    res = jax.vmap(partial(advance, cfg))(jax.random.split(k_step, n), states, prey_act, pred_act)

    def reseed(s):
        k_thin, k_worlds = jax.random.split(k_reset)
        pool = thin_plants(k_thin, gene_pool(s))
        fresh = jax.vmap(partial(reset, cfg), in_axes=(0, None))(jax.random.split(k_worlds, n), pool)
        return jax.tree.map(lambda a, b: jnp.where(res.done.reshape((n,) + (1,) * (a.ndim - 1)), a, b), fresh, s)

    next_state = jax.lax.cond(res.done.any(), reseed, lambda s: s, res.state)
    prey_obs, pred_obs = jax.vmap(partial(observe, cfg))(next_state)
    return res._replace(state=next_state), prey_obs, pred_obs


def extra_metrics(info: StepInfo) -> dict:
    """Gene means and spreads over living agent-steps (and plant cell-steps)."""
    out = {}

    def stats(prefix, names, total, sq, count):
        count = jnp.maximum(count.sum().astype(jnp.float32), 1.0)
        total = total.reshape(-1, len(names)).sum(0)
        sq = sq.reshape(-1, len(names)).sum(0)
        mean = total / count
        std = jnp.sqrt(jnp.maximum(sq / count - mean ** 2, 0.0))
        for i, name in enumerate(names):
            out[f"{prefix}_{name}"] = mean[i]
            out[f"{prefix}_{name}_std"] = std[i]
        return count

    n_prey = stats("prey", GENE_NAMES["animal"], info.prey_gene_sum, info.prey_gene_sq, info.prey_alive)
    n_pred = stats("pred", GENE_NAMES["animal"], info.pred_gene_sum, info.pred_gene_sq, info.pred_alive)
    stats("plant", GENE_NAMES["plant"], info.plant_gene_sum, info.plant_gene_sq, info.plant_count)
    out["prey_gen"] = info.prey_gen_sum.sum() / n_prey
    out["pred_gen"] = info.pred_gen_sum.sum() / n_pred
    out["plant_cover"] = info.plant_cover.mean()
    out["burrow_use_near"] = info.prey_near_burrow.sum() / jnp.maximum(info.prey_near.sum(), 1)
    out["burrow_use_far"] = info.prey_far_burrow.sum() / jnp.maximum(info.prey_far.sum(), 1)
    out["good_bite_share"] = info.good_bites.sum() / jnp.maximum((info.good_bites + info.bad_bites).sum(), 1)
    out["fat_selectivity"] = info.eaten_energy.sum() / jnp.maximum(info.eaten_baseline.sum(), 1e-6)
    out["pred_plant_share"] = info.pred_plant_gain.sum() / jnp.maximum((info.pred_plant_gain + info.pred_meat_gain).sum(), 1e-6)
    return out


def reset_mixed(cfg: EvoConfig, key: jax.Array, pool: GenePool, genes=None) -> State:
    """A fresh world for evaluation: a random half of each species' founders
    carry pooled (evolved) values for the genes listed in `genes` and tag 1;
    the rest carry founder genes and tag 0. Plants come from the pool."""
    k_evo, k_found, k_prey, k_pred = jax.random.split(key, 4)
    evolved = reset(cfg, k_evo, pool)
    founders = reset(cfg, k_found)
    genes = tuple(range(ANIMAL_GENES)) if genes is None else genes  # None: every gene
    use = jnp.isin(jnp.arange(ANIMAL_GENES), jnp.asarray(genes))

    def mix(k, e: Species, f: Species, n_alive: int) -> Species:
        # Tagged founders take random slots, since lower slots win ties in grazing and breeding.
        score = jnp.where(e.alive, jax.random.uniform(k, e.alive.shape), jnp.inf)
        position = jnp.argsort(jnp.argsort(score))  # each slot's rank among living founders
        tagged = e.alive & (position < n_alive // 2)
        mixed = jnp.where(use[None, :], e.genes, f.genes)
        return e._replace(
            genes=jnp.where(tagged[:, None], mixed, f.genes),
            gen=jnp.where(tagged, e.gen, 0),
            tag=tagged.astype(jnp.int32),
        )

    return evolved._replace(prey=mix(k_prey, evolved.prey, founders.prey, cfg.init_prey),
                            pred=mix(k_pred, evolved.pred, founders.pred, cfg.init_pred))


def thin_plants(key: jax.Array, pool: GenePool, n: int = POOL_SAMPLE) -> GenePool:
    """Replace the plant pool (one row per cell of every world) with n living
    plants drawn from it, so resets draw from a small pool. An empty pool stays empty."""
    total = pool.plant_w.sum()
    p = jnp.where(total > 0, pool.plant_w / jnp.maximum(total, 1e-9), 1.0 / pool.plant_w.shape[0])
    idx = jax.random.choice(key, pool.plant_w.shape[0], (n,), p=p)
    return pool._replace(plant_genes=pool.plant_genes[idx], plant_w=jnp.where(total > 0, 1.0, 0.0) * jnp.ones(n))



def pool_to_arrays(pool: GenePool, seed: int = 0) -> dict:
    """Living genomes as numpy arrays, each kind subsampled to POOL_SAMPLE."""
    rng = np.random.default_rng(seed)

    def keep(*arrays, w):
        live = np.nonzero(np.asarray(w) > 0)[0]
        if live.size > POOL_SAMPLE:
            live = np.sort(rng.choice(live, POOL_SAMPLE, replace=False))
        return [np.asarray(a)[live] for a in arrays]

    prey_genes, prey_gen = keep(pool.prey_genes, pool.prey_gen, w=pool.prey_w)
    pred_genes, pred_gen = keep(pool.pred_genes, pool.pred_gen, w=pool.pred_w)
    (plant_genes,) = keep(pool.plant_genes, w=pool.plant_w)
    return {"prey_genes": prey_genes, "prey_gen": prey_gen, "pred_genes": pred_genes,
            "pred_gen": pred_gen, "plant_genes": plant_genes}


def pad_genes(genes, width: int) -> np.ndarray:
    """(n, w) genes as (n, width): genes saved before size (w = 3) get FOUNDER_SIZE."""
    genes = np.asarray(genes, np.float32)
    genes = genes.reshape(len(genes), -1) if genes.size else np.zeros((0, width), np.float32)
    if genes.shape[1] < width:
        genes = np.concatenate([genes, np.full((len(genes), width - genes.shape[1]), FOUNDER_SIZE, np.float32)], 1)
    return genes


def pool_from_arrays(d) -> GenePool:
    """Inverse of pool_to_arrays. An empty kind becomes one zero-weight row, so
    draws from it fall back to founders."""
    def kind(genes, width, gen=None):
        genes = pad_genes(genes, width)
        if len(genes) == 0:
            g = jnp.zeros((1, width))
            return g, jnp.zeros(1, jnp.int32), jnp.zeros(1)
        gen = np.zeros(len(genes), np.int32) if gen is None else np.asarray(gen, np.int32)
        return jnp.asarray(genes), jnp.asarray(gen), jnp.ones(len(genes))

    prey = kind(d["prey_genes"], ANIMAL_GENES, d["prey_gen"])
    pred = kind(d["pred_genes"], ANIMAL_GENES, d["pred_gen"])
    plant_genes, _, plant_w = kind(d["plant_genes"], PLANT_GENES)
    return GenePool(*prey, *pred, plant_genes, plant_w)
