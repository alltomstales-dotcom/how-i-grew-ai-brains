import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_checkpoint, save_checkpoint
from agents.networks import ActorCritic, init_params
from agents.policies import network_policy, random_policy


def test_actor_critic_shapes():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jnp.zeros((3, 7, 9, 9, 4), jnp.bfloat16)
    logits, value = net.apply(params, obs)
    assert logits.shape == (3, 7, 5) and value.shape == (3, 7)
    assert logits.dtype == jnp.float32


def test_initial_policy_is_near_uniform():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (16, 9, 9, 4))
    logits, _ = net.apply(params, obs)
    probs = np.asarray(jax.nn.softmax(logits))
    assert np.all(np.abs(probs - 0.2) < 0.05)


def test_random_policy_range_and_shape():
    a = random_policy(jax.random.PRNGKey(0), jnp.zeros((4, 6, 9, 9, 4)))
    assert a.shape == (4, 6)
    assert int(a.min()) >= 0 and int(a.max()) <= 4


def test_network_policy_samples_valid_actions():
    net = ActorCritic()
    policy = network_policy(net, init_params(net, jax.random.PRNGKey(0), 9))
    a = policy(jax.random.PRNGKey(2), jnp.zeros((4, 6, 9, 9, 4)))
    assert a.shape == (4, 6)
    assert int(a.min()) >= 0 and int(a.max()) <= 4


def test_checkpoint_roundtrip(tmp_path):
    net = ActorCritic()
    prey = init_params(net, jax.random.PRNGKey(0), 9)
    pred = init_params(net, jax.random.PRNGKey(1), 9)
    path = tmp_path / "ckpt" / "c.msgpack"
    save_checkpoint(path, prey, pred, {"update": 7, "env_steps": 1234})
    template = init_params(net, jax.random.PRNGKey(99), 9)
    prey2, pred2, meta = load_checkpoint(path, template)
    for a, b in zip(jax.tree.leaves(prey), jax.tree.leaves(prey2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree.leaves(pred), jax.tree.leaves(pred2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert meta == {"update": 7, "env_steps": 1234}
    assert not list(path.parent.glob("*.tmp"))
