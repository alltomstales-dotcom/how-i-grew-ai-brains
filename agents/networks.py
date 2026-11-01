"""Policy/value network shared in shape by both species (each has its own params)."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from simulations.predprey import NUM_ACTIONS, OBS_CHANNELS


class ActorCritic(nn.Module):
    """memory=0: feed-forward, (obs) -> (logits, value). memory=H: a GRU of size H,
    (obs, h) -> (logits, value, h'); its output joins the trunk through a zero-initialised
    projection, so feed-forward weights loaded into it act exactly as before."""
    num_actions: int = NUM_ACTIONS
    memory: int = 0
    tiny: bool = False  # about 4,700 weights rather than 115,750, so each animal can hold its own (Stage 14a)
    brain: bool = False  # the last input channel is a brain gene g: only the first 8 + round(120 g) hidden units are used

    @nn.compact
    def __call__(self, obs: jax.Array, h: jax.Array | None = None):
        lead = obs.shape[:-3]
        x = obs.reshape((-1,) + obs.shape[-3:]).astype(jnp.float32)
        g = x[:, 0, 0, -1]
        width, hidden = (8, 16) if self.tiny else (32, 128)
        x = nn.relu(nn.Conv(width, (3, 3), padding="VALID")(x))
        x = nn.relu(nn.Conv(width, (3, 3), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(hidden)(x))
        if self.brain:
            x = x * (jnp.arange(128)[None, :] < 8 + jnp.round(120 * g)[:, None])
        if self.memory:
            h, _ = nn.GRUCell(self.memory, name="gru")(h.reshape((-1, self.memory)).astype(jnp.float32), x)
            x = x + nn.Dense(128, kernel_init=nn.initializers.zeros, name="mem_proj")(h)
        logits = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[:, 0]
        out = logits.reshape(lead + (self.num_actions,)), value.reshape(lead)
        return out + (h.reshape(lead + (self.memory,)),) if self.memory else out


def init_params(net: ActorCritic, key: jax.Array, view: int = 9, channels: int = OBS_CHANNELS):
    obs = jnp.zeros((1, view, view, channels), jnp.float32)
    return net.init(key, obs, jnp.zeros((1, net.memory))) if net.memory else net.init(key, obs)
