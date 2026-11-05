"""Save and load both species' parameters in one msgpack file."""
from __future__ import annotations

from pathlib import Path

from flax import serialization


def save_checkpoint(path, prey_params, pred_params, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = serialization.msgpack_serialize(
        {
            "prey": serialization.to_state_dict(prey_params),
            "pred": serialization.to_state_dict(pred_params),
            "meta": dict(meta),
        }
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_checkpoint(path, template):
    raw = serialization.msgpack_restore(Path(path).read_bytes())
    prey = serialization.from_state_dict(template, raw["prey"])
    pred = serialization.from_state_dict(template, raw["pred"])
    return prey, pred, dict(raw["meta"])


def widen_actions(sd, full) -> None:
    """Stage 21: a brain saved with 5 actions, loaded into a world with 10 (each move with a call off or on). Every layer
    whose outputs are a whole multiple of the saved one's is tiled, so moves a and a + 5 get the same logit: the chance
    of each move is exactly as before, and the call is on half the time, whatever the animal sees."""
    import numpy as np

    for name, layer in full["params"].items():
        have = sd["params"].get(name)
        if not have or "kernel" not in have or "kernel" not in layer:
            continue
        k_have, k_want = np.asarray(have["kernel"]), np.asarray(layer["kernel"])
        if k_have.ndim != 2 or k_have.shape == k_want.shape or k_want.shape[0] != k_have.shape[0]:
            continue
        reps, rem = divmod(k_want.shape[1], k_have.shape[1])
        if rem or reps < 2:
            raise ValueError(f"layer {name}: can't widen {k_have.shape} to {k_want.shape}")
        have["kernel"] = np.tile(k_have, (1, reps))
        have["bias"] = np.tile(np.asarray(have["bias"]), reps)


def load_brains(path, template):
    """Load both species into `template`'s shape. When the template sees more input
    channels than the checkpoint did, the first convolution gets zero weights for the
    extra (last) channels, so the brain acts exactly as before until they light up.
    A brain saved with fewer actions is widened by widen_actions."""
    import numpy as np

    raw = serialization.msgpack_restore(Path(path).read_bytes())
    full = serialization.to_state_dict(template)
    want = full["params"]["Conv_0"]["kernel"].shape[2]

    def adapt(sd):
        kernel = np.asarray(sd["params"]["Conv_0"]["kernel"])
        have = kernel.shape[2]
        if have > want:
            raise ValueError(f"checkpoint sees {have} input channels; this world gives only {want}")
        pad = np.zeros(kernel.shape[:2] + (want - have, kernel.shape[3]), kernel.dtype)
        sd["params"]["Conv_0"]["kernel"] = np.concatenate([kernel, pad], axis=2)
        widen_actions(sd, full)
        for name, layer in full["params"].items():  # layers the saved brain lacks (e.g. memory) start fresh
            sd["params"].setdefault(name, layer)
        return serialization.from_state_dict(template, sd)

    return adapt(raw["prey"]), adapt(raw["pred"]), dict(raw["meta"])
