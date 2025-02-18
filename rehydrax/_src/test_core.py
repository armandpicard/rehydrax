import jax
import jax.numpy as jnp
import numpy as np

import pytest
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as mlir_interpreter
import rehydrax


def generic_test_jax_function(f, inputs):
    """Test that lowering a jax functions to stablehle and then reloading it give the same result."""
    y_expected = f(*inputs)
    f_lowered = jax.jit(f).lower(*inputs)
    ctx = mlir_interpreter.make_ir_context()
    with ctx:
        m = ir.Module.parse(f_lowered.as_text())
        f_rehydrated = rehydrax.rehydrate_stablehlo_module(m)

        y_rehydrated = f_rehydrated(*inputs)
        assert np.all(y_expected == y_rehydrated)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x + 1,
        lambda x: -x,
        jax.lax.sign,
        jax.lax.floor,
        jax.lax.ceil,
        jax.lax.is_finite,
        jax.lax.exp,
        jax.lax.exp2,
        jax.lax.expm1,
        jax.lax.log,
        jax.lax.log1p,
        jax.lax.tanh,
        jax.lax.logistic,
        jax.lax.sin,
        jax.lax.cos,
        jax.lax.abs,
        jax.lax.sqrt,
        jax.lax.rsqrt,
        jax.lax.cbrt,
    ],
)
def test_unary_functions_float(f):
    generic_test_jax_function(f, (jnp.array(1.0),))


def test_select():
    def f(x, y, z):
        return jax.lax.select(x > 0, y, z)

    generic_test_jax_function(f, (jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)))
