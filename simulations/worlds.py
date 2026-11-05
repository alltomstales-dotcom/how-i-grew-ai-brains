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
    "lasting": (evo, evo.LastingConfig),
    "bulk_base": (evo, evo.BulkBaseConfig),
    "bulk_cheap": (evo, evo.BulkCheapConfig),
    "bulk_nocatch": (evo, evo.BulkNoCatchConfig),
    "bulk_certain": (evo, evo.BulkCertainConfig),
    "bulk_flat": (evo, evo.BulkFlatConfig),
    "bulk_coin": (evo, evo.BulkCoinConfig),
    "bulk_rich": (evo, evo.BulkRichConfig),
    "bulk_certain_rich": (evo, evo.BulkCertainRichConfig),
    "bulk_coin_rich": (evo, evo.BulkCoinRichConfig),
    "bulk_cap_coin": (evo, evo.BulkCapCoinConfig),
    "bulk_cap_burn": (evo, evo.BulkCapBurnConfig),
    "bulk_cap_free": (evo, evo.BulkCapFreeConfig),
    "bulk_cap_rich": (evo, evo.BulkCapRichConfig),
    "bulk_feast_50": (evo, evo.BulkFeast50Config),
    "bulk_feast_70": (evo, evo.BulkFeast70Config),
    "bulk_feast_meadow": (evo, evo.BulkFeastMeadowConfig),
    "bulk_feast_forest": (evo, evo.BulkFeastForestConfig),
    "bulk_feast_tundra": (evo, evo.BulkFeastTundraConfig),
    "bulk_winter": (evo, evo.BulkWinterConfig),
    "bulk_winter_free": (evo, evo.BulkWinterFreeConfig),
    "bulk_winter_flat": (evo, evo.BulkWinterFlatConfig),
    "camo": (evo, evo.CamoConfig),
    "camo_free": (evo, evo.CamoFreeConfig),
    "ambush_stealth": (evo, evo.AmbushStealthConfig),
    "brain_big": (evo, evo.BrainBigConfig),
    "tiny_stealth": (evo, evo.TinyStealthConfig),
    "tiny_winter": (evo, evo.TinyWinterConfig),
    "camo_winter_tundra": (evo, evo.CamoWinterTundraConfig),
    "camo_half_tundra": (evo, evo.CamoHalfTundraConfig),
    "mut_winter_tundra": (evo, evo.MutWinterTundraConfig),
    "mut_zone_tundra": (evo, evo.MutZoneTundraConfig),
    "mut_even_tundra": (evo, evo.MutEvenTundraConfig),
    "mut_random_tundra": (evo, evo.MutRandomTundraConfig),
    "camo_neutral": (evo, evo.CamoNeutralConfig),
    "camo_meadow": (evo, evo.CamoMeadowConfig),
    "camo_forest": (evo, evo.CamoForestConfig),
    "camo_tundra": (evo, evo.CamoTundraConfig),
    "camo_start": (evo, evo.CamoStartConfig),
    "camo_10": (evo, evo.Camo10Config),
    "camo_05": (evo, evo.Camo05Config),
    "bulk_meadow": (evo, evo.BulkMeadowConfig),
    "bulk_forest": (evo, evo.BulkForestConfig),
    "bulk_tundra": (evo, evo.BulkTundraConfig),
    "meadow": (evo, evo.ContinentConfig),
    "forest": (evo, evo.ForestConfig),
    "tundra": (evo, evo.TundraConfig),
    "lasting_room": (evo, evo.LastingRoomConfig),
    "lasting_stealth": (evo, evo.LastingStealthConfig),
    "lasting_stealth_m100": (evo, evo.LastingStealthM100Config),
    "lasting_sprint": (evo, evo.LastingSprintConfig),
    "lasting_sprint_m100": (evo, evo.LastingSprintM100Config),
    "lasting_ambush": (evo, evo.LastingAmbushConfig),
    "lasting_ambush_m100": (evo, evo.LastingAmbushM100Config),
    "lasting_m100": (evo, evo.LastingM100Config),
    "lasting_m250": (evo, evo.LastingM250Config),
    "omni": (evo, evo.OmniConfig),
    "omni_tough": (evo, evo.OmniToughConfig),
    "omni_neutral": (evo, evo.OmniNeutralConfig),
    "omni_d03": (evo, evo.OmniD03Config),
    "omni_d08": (evo, evo.OmniD08Config),
    "omni_d20": (evo, evo.OmniD20Config),
    "omni_d11": (evo, evo.OmniD11Config),
    "omni_d11_lean": (evo, evo.OmniD11LeanConfig),
    "omni_d14": (evo, evo.OmniD14Config),
    "omni_winter": (evo, evo.OmniWinterConfig),
    "fatprey": (evo, evo.FatPreyConfig),
    "fatprey_rich": (evo, evo.FatPreyRichConfig),
    "fatprey_fixed": (evo, evo.FatPreyFixedConfig),
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
    return ActorCritic(num_actions=num_actions(world, cfg), memory=memory, brain=getattr(cfg, "brains", False),
                       tiny=getattr(cfg, "tiny_brain", False), scale=getattr(cfg, "brain_scale", 1))


def world_set(cfg, pairs):
    """Stage 20: a config with fields changed from "key=value" strings (repeatable --world-set), each value read as its field's type.
    For example ["pred_base_cost=0.1", "gene_sigma=0"]. An unknown field is an error, so a typo can't silently run the usual world."""
    import dataclasses
    import json
    changes = {}
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not hasattr(cfg, key):
            raise ValueError(f"--world-set {pair!r}: expected key=value with a field of {type(cfg).__name__}")
        old, raw = getattr(cfg, key), raw.strip()
        if isinstance(old, bool):
            if raw.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"--world-set {pair!r}: {key} is true or false")
            value = raw.lower() in ("true", "1")
        elif isinstance(old, int):
            value = int(raw)
        elif isinstance(old, float):
            value = float(raw)
        elif isinstance(old, tuple):
            value = tuple(json.loads(raw))
        else:
            value = raw
        changes[key] = value
    return dataclasses.replace(cfg, **changes) if changes else cfg
