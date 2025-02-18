"""Exmaple showing how to export and import a JAX model using stablehlo."""

import jax
from flax import nnx
import numpy as np
import jax.numpy as jnp
import jax.core
import rehydrax


class Model(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(2, 3, rngs=rngs)
        self.linear2 = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


def main():
    #############
    # Project A #
    #############
    @jax.jit
    def init(rng):
        model = Model(rngs=nnx.Rngs(rng))
        return nnx.split(model)[1]

    @jax.jit
    def forward(state, x):
        model = Model(nnx.Rngs(0))
        model_graph = nnx.split(model)[0]
        model = nnx.merge(model_graph, state)
        return model(x)

    key = jax.random.PRNGKey(0)

    init_lowered = init.lower(key)
    init_stablehlo = init_lowered.as_text()
    state_1 = init(key)

    sample = jax.random.uniform(key, (1, 2), jnp.float32)
    forward_lowered = forward.lower(state_1, sample)
    forward_stablehlo = forward_lowered.as_text()
    y1 = forward(state_1, sample)

    #############
    # Project B #
    #############
    # rehydrate and run init
    init_rehydrated = rehydrax.rehydrate_stablehlo(init_stablehlo)
    state_2 = init_rehydrated(key)

    # Compare state to project A
    state_1_flat = jax.tree.flatten(state_1)[0]
    state_2_flat = jax.tree.flatten(state_2)[0]
    for a, b in zip(state_1_flat, state_2_flat):
        assert np.allclose(a, b)

    # rehydrate and run forward
    forward_rehydrated = rehydrax.rehydrate_stablehlo(forward_stablehlo)
    y2 = forward_rehydrated(*state_2, sample)

    # Compare y to project A
    assert np.allclose(y1, y2)


if __name__ == "__main__":
    main()
