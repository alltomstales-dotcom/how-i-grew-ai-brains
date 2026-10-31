"""PPO for two species at once, in the style of PureJaxRL.

Each agent slot is an independent actor. A slot's trajectory ends when its
agent dies, when the slot was empty, or when the episode ends. Transitions
from empty slots are masked out of the loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from agents.networks import ActorCritic, init_params
from simulations import predprey as pp
from simulations.worlds import get_world, make_net, obs_channels


@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 512  # 1024 ran out of memory under a DGX Spark's ~12 GB XLA cap
    rollout_len: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    num_minibatches: int = 16
    seed: int = 0
    learn: bool = True  # False = control run: brains stay at their random start
    learners: str = "both"  # "prey" or "pred": only that species learns; the other's brain stays as it started
    sample_frac: float = 1.0  # share of agent-slots each update learns from, living animals first
    league_size: int = 0  # frozen past brains kept per species (0 = no league)
    league_frac: float = 0.25  # worlds per side that face a past brain: first past predators, then past prey
    memory: int = 0  # GRU size per brain (0 = feed-forward)


def pick_samples(key: jax.Array, mask: jax.Array, k: int) -> jax.Array:
    """k indices into a flat batch: living (mask) slots first, in random order; a random
    subset of them when more than k are alive, padded with empty slots when fewer."""
    score = mask.astype(jnp.float32) + 0.5 * jax.random.uniform(key, mask.shape)
    return jnp.argsort(-score)[:k]


def tiny_configs(world_name: str = "predprey") -> tuple[pp.WorldConfig, TrainConfig]:
    """Small enough to train a few updates on a laptop CPU in seconds."""
    _, config_cls = get_world(world_name)
    return (
        config_cls(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40),
        TrainConfig(num_envs=4, rollout_len=8, num_minibatches=2, epochs=1),
    )


class Learner(NamedTuple):
    params: Any
    opt_state: Any


class SpeciesBatch(NamedTuple):
    obs: jax.Array  # (T, E, N, view, view, C)
    action: jax.Array  # (T, E, N)
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array  # bool
    mask: jax.Array  # bool: the slot held a living agent when it acted
    keep: Any = None  # memory brains: bool, the slot's memory carries into the next step


class League(NamedTuple):
    prey: Any  # params stacked on a leading axis of league_size
    pred: Any
    count: jax.Array  # snapshots added so far (ring buffer)


class Runner(NamedTuple):
    prey: Learner
    pred: Learner
    env_state: Any  # the world's State, batched over envs
    prey_obs: jax.Array
    pred_obs: jax.Array
    key: jax.Array
    league: Any = None  # League, or None without one
    prey_h: Any = None  # memory brains: (E, N, H) per-slot memory
    pred_h: Any = None


class Trainer(NamedTuple):
    net: ActorCritic
    init: Callable[[jax.Array], Runner]
    update: Callable[[Runner], tuple[Runner, dict]]
    add_to_league: Callable[[Runner], Runner] = lambda runner: runner


def league_keep(num_envs: int, frac: float, active) -> tuple[jax.Array, jax.Array]:
    """Per-world masks of whose actions each species learns from. With a league active,
    worlds [0, m) have past predators and worlds [m, 2m) past prey, m = num_envs * frac;
    a frozen side is no one's training data."""
    m = int(num_envs * frac)
    e = jnp.arange(num_envs)
    pred_keep = ~(active & (e < m))
    prey_keep = ~(active & (e >= m) & (e < 2 * m))
    return prey_keep, pred_keep


def compute_gae(reward, value, done, last_value, gamma, lam):
    """Generalised advantage estimation over a time-major rollout.
    done[t] stops bootstrapping from step t + 1."""
    not_done = 1.0 - done.astype(jnp.float32)

    def body(carry, x):
        gae, next_value = carry
        r, v, nd = x
        delta = r + gamma * next_value * nd - v
        gae = delta + gamma * lam * nd * gae
        return (gae, v), gae

    _, adv = jax.lax.scan(
        body, (jnp.zeros_like(last_value), last_value), (reward, value, not_done), reverse=True
    )
    return adv, adv + value


def masked_normalize(x, mask):
    m = mask.astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    mean = (x * m).sum() / n
    var = (((x - mean) ** 2) * m).sum() / n
    return (x - mean) / jnp.sqrt(var + 1e-8)


def ppo_loss(params, mb: dict, apply_fn, cfg: TrainConfig):
    logits, value = apply_fn(params, mb["obs"])
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, mb["action"][..., None], axis=-1)[..., 0]
    m = mb["mask"].astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    ratio = jnp.exp(logp - mb["logp"])
    adv = mb["adv"]
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
    pg_loss = (pg * m).sum() / n
    v_loss = 0.5 * (((value - mb["ret"]) ** 2) * m).sum() / n
    entropy = ((-(jnp.exp(logp_all) * logp_all).sum(-1)) * m).sum() / n
    total = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    return total, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}


def rollout_metrics(info: pp.StepInfo, prey_b: SpeciesBatch, pred_b: SpeciesBatch) -> dict:
    steps = info.catches.size  # T * E world-steps

    def total(x):
        return x.sum().astype(jnp.float32)

    def ratio(num, den):
        return total(num) / jnp.maximum(total(den), 1.0)

    return {
        "prey_alive": total(info.prey_alive) / steps,
        "pred_alive": total(info.pred_alive) / steps,
        "catches_per_step": total(info.catches) / steps,
        "prey_births_per_step": total(info.prey_births) / steps,
        "pred_births_per_step": total(info.pred_births) / steps,
        "prey_deaths_per_step": total(info.prey_deaths) / steps,
        "pred_deaths_per_step": total(info.pred_deaths) / steps,
        "prey_pred_dist": ratio(info.prey_pred_dist_sum, info.prey_pred_dist_count),
        "pred_cluster_dist": ratio(info.pred_pred_dist_sum, info.pred_pred_dist_count),
        "prey_reward": ratio(prey_b.reward * prey_b.mask, prey_b.mask),
        "pred_reward": ratio(pred_b.reward * pred_b.mask, pred_b.mask),
    }


def make_trainer(wcfg: pp.WorldConfig, tcfg: TrainConfig, world=None) -> Trainer:
    """`world` is a world module (see simulations/worlds.py); predprey by default."""
    world = world or pp
    E, T = tcfg.num_envs, tcfg.rollout_len
    for n in (wcfg.max_prey, wcfg.max_pred):
        if (T * E * n) % tcfg.num_minibatches:
            raise ValueError(
                f"rollout_len*num_envs*{n} = {T * E * n} is not divisible by "
                f"num_minibatches={tcfg.num_minibatches}"
            )

    if tcfg.memory and getattr(wcfg, "brains", False):
        raise ValueError("memory brains with an evolving brain size are not supported yet")
    if tcfg.memory and tcfg.league_size:
        raise ValueError("memory brains with a league are not supported yet")
    if tcfg.learners not in ("both", "prey", "pred"):
        raise ValueError(f"learners must be both, prey or pred, not {tcfg.learners!r}")
    H = tcfg.memory
    net = make_net(world, wcfg, H)
    tx = optax.chain(optax.clip_by_global_norm(tcfg.max_grad_norm), optax.adam(tcfg.lr, eps=1e-5))
    reset_v = jax.vmap(partial(world.reset, wcfg))
    observe_v = jax.vmap(partial(world.observe, wcfg))
    loss_fn = partial(ppo_loss, apply_fn=net.apply, cfg=tcfg)

    def init(key):
        k_env, k_prey, k_pred, key = jax.random.split(key, 4)
        env_state = reset_v(jax.random.split(k_env, E))
        prey_obs, pred_obs = observe_v(env_state)
        prey_params = init_params(net, k_prey, wcfg.view, obs_channels(world, wcfg))
        pred_params = init_params(net, k_pred, wcfg.view, obs_channels(world, wcfg))
        league = None
        if tcfg.league_size:
            stack = lambda p: jax.tree.map(lambda x: jnp.broadcast_to(x, (tcfg.league_size,) + x.shape), p)
            league = League(stack(prey_params), stack(pred_params), jnp.int32(0))
        return Runner(
            Learner(prey_params, tx.init(prey_params)),
            Learner(pred_params, tx.init(pred_params)),
            env_state,
            prey_obs,
            pred_obs,
            key,
            league,
            jnp.zeros((E, wcfg.max_prey, H)) if H else None,
            jnp.zeros((E, wcfg.max_pred, H)) if H else None,
        )

    def act(params, key, obs, h=None):
        if H:
            logits, value, h = net.apply(params, obs, h)
        else:
            logits, value = net.apply(params, obs)
        action = jax.random.categorical(key, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]
        return action, logp, value, h

    m = int(E * tcfg.league_frac)

    def env_step(runner, past):
        key, k_prey, k_pred, k_env = jax.random.split(runner.key, 4)
        prey_a, prey_logp, prey_v, prey_h = act(runner.prey.params, k_prey, runner.prey_obs, runner.prey_h)
        pred_a, pred_logp, pred_v, pred_h = act(runner.pred.params, k_pred, runner.pred_obs, runner.pred_h)
        prey_keep = pred_keep = jnp.ones(E, bool)
        if past is not None:  # past predators in worlds [0, m), past prey in [m, 2m)
            past_prey, past_pred, active = past
            old_pred = act(past_pred, k_pred, runner.pred_obs[:m])[0]  # no league with memory brains
            old_prey = act(past_prey, k_prey, runner.prey_obs[m:2 * m])[0]
            pred_a = pred_a.at[:m].set(jnp.where(active, old_pred, pred_a[:m]))
            prey_a = prey_a.at[m:2 * m].set(jnp.where(active, old_prey, prey_a[m:2 * m]))
            prey_keep, pred_keep = league_keep(E, tcfg.league_frac, active)
        res, prey_obs, pred_obs = world.batch_step(wcfg, k_env, runner.env_state, prey_a, pred_a)
        prey_carry = pred_carry = None
        if H:  # memory survives only while the same animal lives on in the same world
            # *_done = died | slot was empty | world ended. A slot alive before and after can still
            # hold a newborn that took a just-freed slot: done says so, alive does not.
            prey_carry = res.state.prey.alive & ~res.prey_done
            pred_carry = res.state.pred.alive & ~res.pred_done
            prey_h = jnp.where(prey_carry[..., None], prey_h, 0.0)
            pred_h = jnp.where(pred_carry[..., None], pred_h, 0.0)
        prey_b = SpeciesBatch(runner.prey_obs, prey_a, prey_logp, prey_v, res.prey_reward, res.prey_done,
                              runner.env_state.prey.alive & prey_keep[:, None], prey_carry)
        pred_b = SpeciesBatch(runner.pred_obs, pred_a, pred_logp, pred_v, res.pred_reward, res.pred_done,
                              runner.env_state.pred.alive & pred_keep[:, None], pred_carry)
        runner = runner._replace(env_state=res.state, prey_obs=prey_obs, pred_obs=pred_obs, key=key,
                                 prey_h=prey_h, pred_h=pred_h)
        return runner, (prey_b, pred_b, res.info)

    def learn_memory(learner, batch: SpeciesBatch, h0, adv, ret, key):
        """Minibatches of whole slot sequences, replaying each slot's memory from h0."""
        seq = {"obs": batch.obs, "action": batch.action, "logp": batch.logp, "mask": batch.mask,
               "keep": batch.keep, "adv": adv, "ret": ret}
        seq = jax.tree.map(lambda x: x.reshape(x.shape[:1] + (-1,) + x.shape[3:]), seq)  # (T, slots, ...)
        h0 = h0.reshape(-1, H)
        slots = h0.shape[0]
        n = slots if tcfg.sample_frac >= 1 else max(int(slots * tcfg.sample_frac) // tcfg.num_minibatches, 1) * tcfg.num_minibatches
        key, k_pick = jax.random.split(key)
        lived = seq["mask"].sum(0).astype(jnp.float32)
        chosen = jnp.argsort(-(lived + 0.5 * jax.random.uniform(k_pick, lived.shape)))[:n]

        def replay(params, obs):
            obs, h, keep = obs

            def body(h, x):
                o, k = x
                logits, value, h_out = net.apply(params, o, h)
                return jnp.where(k[:, None], h_out, 0.0), (logits, value)

            _, (logits, value) = jax.lax.scan(body, h, (obs, keep))
            return logits, value

        loss = partial(ppo_loss, apply_fn=replay, cfg=tcfg)

        def epoch(learner, k):
            idx = chosen[jax.random.permutation(k, n)].reshape(tcfg.num_minibatches, -1)

            def minibatch(learner, i):
                mb = {name: x[:, i] for name, x in seq.items()}
                mb["obs"] = (mb["obs"], h0[i], mb.pop("keep"))
                (_, aux), grads = jax.value_and_grad(loss, has_aux=True)(learner.params, mb)
                updates, opt_state = tx.update(grads, learner.opt_state, learner.params)
                return Learner(optax.apply_updates(learner.params, updates), opt_state), aux

            return jax.lax.scan(minibatch, learner, idx)

        learner, aux = jax.lax.scan(epoch, learner, jax.random.split(key, tcfg.epochs))
        return learner, jax.tree.map(jnp.mean, aux)

    def learn(learner, batch: SpeciesBatch, last_value, key, h0=None):
        adv, ret = compute_gae(batch.reward, batch.value, batch.done, last_value, tcfg.gamma, tcfg.gae_lambda)
        adv = masked_normalize(adv, batch.mask)
        if H:
            return learn_memory(learner, batch, h0, adv, ret, key)
        flat = {"obs": batch.obs, "action": batch.action, "logp": batch.logp,
                "mask": batch.mask, "adv": adv, "ret": ret}
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), flat)
        total = flat["action"].shape[0]
        # Most slots hold no living animal; learning from all of them made PPO the bottleneck.
        n = total if tcfg.sample_frac >= 1 else max(int(total * tcfg.sample_frac) // tcfg.num_minibatches, 1) * tcfg.num_minibatches
        key, k_pick = jax.random.split(key)
        chosen = jnp.arange(total) if n == total else pick_samples(k_pick, flat["mask"], n)

        def epoch(learner, k):
            # Gather one minibatch at a time: shuffling the whole batch at once copied
            # it, and at 1024 worlds that copy alone was 11.7 GiB.
            idx = chosen[jax.random.permutation(k, n)].reshape(tcfg.num_minibatches, -1)

            def minibatch(learner, i):
                mb = jax.tree.map(lambda x: x[i], flat)
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(learner.params, mb)
                updates, opt_state = tx.update(grads, learner.opt_state, learner.params)
                return Learner(optax.apply_updates(learner.params, updates), opt_state), aux

            return jax.lax.scan(minibatch, learner, idx)

        learner, aux = jax.lax.scan(epoch, learner, jax.random.split(key, tcfg.epochs))
        return learner, jax.tree.map(jnp.mean, aux)

    def update(runner):
        past = None
        if runner.league is not None:
            lg = runner.league
            k_pick, key = jax.random.split(runner.key)
            runner = runner._replace(key=key)
            i = jax.random.randint(k_pick, (), 0, jnp.maximum(jnp.minimum(lg.count, tcfg.league_size), 1))
            take = lambda p: jax.tree.map(lambda x: x[i], p)
            past = (take(lg.prey), take(lg.pred), lg.count > 0)
        prey_h0, pred_h0 = runner.prey_h, runner.pred_h
        runner, (prey_b, pred_b, info) = jax.lax.scan(lambda r, _: env_step(r, past), runner, None, length=T)
        prey_last = act(runner.prey.params, runner.key, runner.prey_obs, runner.prey_h)[2]
        pred_last = act(runner.pred.params, runner.key, runner.pred_obs, runner.pred_h)[2]
        key, k_prey, k_pred = jax.random.split(runner.key, 3)
        metrics = rollout_metrics(info, prey_b, pred_b)
        metrics.update(world.extra_metrics(info))
        if past is not None:
            # Catches per predator-step: past predators vs live prey, live predators vs past prey, live vs live.
            def rate(lo, hi):
                return (info.catches[:, lo:hi].sum() / jnp.maximum(info.pred_alive[:, lo:hi].sum(), 1)).astype(jnp.float32)
            metrics.update(league_pred_catch=rate(0, m), league_prey_caught=rate(m, 2 * m), self_catch=rate(2 * m, E),
                           league_count=runner.league.count.astype(jnp.float32))
        if not tcfg.learn:
            return runner._replace(key=key), metrics
        prey, pred = runner.prey, runner.pred
        if tcfg.learners in ("both", "prey"):
            prey, prey_aux = learn(runner.prey, prey_b, prey_last, k_prey, prey_h0)
            metrics.update({f"prey_{k}": v for k, v in prey_aux.items()})
        if tcfg.learners in ("both", "pred"):
            pred, pred_aux = learn(runner.pred, pred_b, pred_last, k_pred, pred_h0)
            metrics.update({f"pred_{k}": v for k, v in pred_aux.items()})
        return runner._replace(prey=prey, pred=pred, key=key), metrics

    def add_to_league(runner):
        lg = runner.league
        slot = lg.count % tcfg.league_size
        put = lambda stack, p: jax.tree.map(lambda s, x: s.at[slot].set(x), stack, p)
        return runner._replace(league=League(put(lg.prey, runner.prey.params), put(lg.pred, runner.pred.params),
                                             lg.count + 1))

    if not tcfg.league_size:
        return Trainer(net, jax.jit(init), jax.jit(update, donate_argnums=0))
    return Trainer(net, jax.jit(init), jax.jit(update, donate_argnums=0), jax.jit(add_to_league))
