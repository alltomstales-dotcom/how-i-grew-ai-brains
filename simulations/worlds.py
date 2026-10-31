"""World registry: name -> (world module, config class).

A world module provides reset(cfg, key), observe(cfg, state),
advance(cfg, key, state, prey_act, pred_act), batch_step(cfg, key, states,
prey_act, pred_act), extra_metrics(info) and OBS_CHANNELS.
"""
from __future__ import annotations

from simulations import evo, predprey

WORLDS = {
    "predprey": (predprey, predprey.WorldConfig),
    "evo": (evo, evo.EvoConfig),
    "terrain": (evo, evo.TerrainConfig),
    "seasons": (evo, evo.SeasonsConfig),
    "niches": (evo, evo.NichesConfig),
    "pack": (evo, evo.PackConfig),
    "signals": (evo, evo.SignalConfig),
    "flank": (evo, evo.FlankConfig),
    "flank_signals": (evo, evo.FlankSignalConfig),
    "pack_brains": (evo, evo.PackBrainsConfig),
    "flank_brains": (evo, evo.FlankBrainsConfig),
    "pack_tax": (evo, evo.PackTaxConfig),
    "flank_tax": (evo, evo.FlankTaxConfig),
    "hills": (evo, evo.HillsConfig),
    "flank_hills": (evo, evo.FlankHillsConfig),
    "ambush": (evo, evo.AmbushConfig),
    "fickle": (evo, evo.FickleConfig),
}


def get_world(name: str):
    return WORLDS[name]


def obs_channels(world, cfg) -> int:
    """Input channels of a brain for this world and config."""
    return world.obs_channels(cfg) if hasattr(world, "obs_channels") else world.OBS_CHANNELS


def num_actions(world, cfg) -> int:
    """Output actions of a brain for this world and config."""
    return world.num_actions(cfg) if hasattr(world, "num_actions") else world.NUM_ACTIONS


def make_net(world, cfg, memory: int = 0):
    """The brain for this world and config: its inputs, actions, memory and (Stage 7a) brain size."""
    from agents.networks import ActorCritic
    return ActorCritic(num_actions=num_actions(world, cfg), memory=memory, brain=getattr(cfg, "brains", False))
