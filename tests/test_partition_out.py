import jax
import jax.numpy as jnp
from jax.core import eval_jaxpr

from taxpr.dfg import partition_out


def test_partition_out_basic():
    """Partition a function returning two values into two single-output jaxprs."""

    def fn(x):
        a = x + 1.0
        b = x * 2.0
        return a, b

    closed = jax.make_jaxpr(fn)(jnp.array(3.0))

    parts = partition_out(closed, [0, 1])

    # Should produce two ClosedJaxprs
    assert len(parts) == 2

    # Original outputs
    orig_out = tuple(eval_jaxpr(closed.jaxpr, closed.consts, jnp.array(3.0)))

    part0_out = tuple(eval_jaxpr(parts[0].jaxpr, parts[0].consts, jnp.array(3.0)))
    part1_out = tuple(eval_jaxpr(parts[1].jaxpr, parts[1].consts, jnp.array(3.0)))

    assert jnp.allclose(part0_out[0], orig_out[0])
    assert jnp.allclose(part1_out[0], orig_out[1])


def test_partition_out_with_constants_and_multiple_inputs():
    """Partition a function that uses multiple inputs and a constant captured in the jaxpr."""

    CONST = 5.0

    def fn(x, y):
        # include a constant use to ensure constvars handled
        a = x + CONST
        b = a * y
        c = b - x
        return b, c

    closed = jax.make_jaxpr(fn)(jnp.array(2.0), jnp.array(3.0))

    parts = partition_out(closed, [0, 1])
    assert len(parts) == 2

    orig_out = tuple(eval_jaxpr(closed.jaxpr, closed.consts, jnp.array(2.0), jnp.array(3.0)))

    out0 = tuple(eval_jaxpr(parts[0].jaxpr, parts[0].consts, jnp.array(2.0), jnp.array(3.0)))
    out1 = tuple(eval_jaxpr(parts[1].jaxpr, parts[1].consts, jnp.array(2.0), jnp.array(3.0)))

    assert jnp.allclose(out0[0], orig_out[0])
    assert jnp.allclose(out1[0], orig_out[1])


def test_partition_out_invalid_index_raises():
    """Providing an outvar index that's out of range should raise IndexError."""

    def fn(x):
        return x + 1.0

    closed = jax.make_jaxpr(fn)(jnp.array(1.0))

    import pytest

    with pytest.raises(IndexError):
        partition_out(closed, [1])
