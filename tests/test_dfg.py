import jax
import jax.numpy as jnp
from jax.core import eval_jaxpr

from taxpr.dfg import partition_out, partial


def test_partition_out_basic():
    """Partition a function returning two values into two single-output jaxprs."""

    def fn(x):
        a = x + 1.0
        b = x * 2.0
        return a, b

    closed = jax.make_jaxpr(fn)(jnp.array(3.0))

    parts = partition_out(closed, [[0], [1]])

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

    parts = partition_out(closed, [[0], [1]])
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
        partition_out(closed, [[1]])


def test_partition_out_with_literal_outputs():
    """Partition a function with literal outputs should handle Literals properly."""

    def fn(x):
        # y is a computed variable, literal_val is a literal constant
        y = x + 1.0
        literal_val = jnp.array(5.0)
        return y, literal_val

    closed = jax.make_jaxpr(fn)(jnp.array(3.0))

    # Should have two outputs: one Var and one Literal
    from jax.extend import core
    assert len(closed.jaxpr.outvars) == 2
    assert isinstance(closed.jaxpr.outvars[0], core.Var)
    assert isinstance(closed.jaxpr.outvars[1], core.Literal)

    # Partition should handle both outputs correctly
    parts = partition_out(closed, [[0], [1]])

    assert len(parts) == 2

    # Original outputs
    orig_out = tuple(eval_jaxpr(closed.jaxpr, closed.consts, jnp.array(3.0)))

    part0_out = tuple(eval_jaxpr(parts[0].jaxpr, parts[0].consts, jnp.array(3.0)))
    part1_out = tuple(eval_jaxpr(parts[1].jaxpr, parts[1].consts, jnp.array(3.0)))

    assert jnp.allclose(part0_out[0], orig_out[0])
    assert jnp.allclose(part1_out[0], orig_out[1])


# Tests for partial function

def test_partial_fix_single_input():
    """Partially evaluate a function by fixing a single input."""

    def fn(x, y):
        return x + y

    closed = jax.make_jaxpr(fn)(jnp.array(2.0), jnp.array(3.0))

    # Fix the first input to 5.0
    partial_closed = partial(closed, {0: 5.0})

    # The partial function should have one input (y)
    assert len(partial_closed.jaxpr.invars) == 1

    # Evaluate the partial function
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(3.0)))

    # Should be 5.0 + 3.0 = 8.0
    assert jnp.allclose(result[0], 8.0)


def test_partial_fix_multiple_inputs():
    """Partially evaluate a function by fixing multiple inputs."""

    def fn(x, y, z):
        return x + y * z

    closed = jax.make_jaxpr(fn)(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0))

    # Fix inputs 0 and 2
    partial_closed = partial(closed, {0: 10.0, 2: 5.0})

    # The partial function should have one input (y)
    assert len(partial_closed.jaxpr.invars) == 1

    # Evaluate the partial function with y = 2.0
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(2.0)))

    # Should be 10.0 + 2.0 * 5.0 = 20.0
    assert jnp.allclose(result[0], 20.0)


def test_partial_no_fixed_inputs():
    """Partially evaluate with no fixed inputs should return a jaxpr with same inputs."""

    def fn(x, y):
        return x * y

    closed = jax.make_jaxpr(fn)(jnp.array(2.0), jnp.array(3.0))
    orig_input_count = len(closed.jaxpr.invars)

    # Fix no inputs
    partial_closed = partial(closed, {})

    # The partial function should have same number of inputs
    assert len(partial_closed.jaxpr.invars) == orig_input_count

    # Evaluate the partial function
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(2.0), jnp.array(3.0)))

    # Should be 2.0 * 3.0 = 6.0
    assert jnp.allclose(result[0], 6.0)


def test_partial_fix_all_inputs():
    """Partially evaluate a function by fixing all inputs."""

    def fn(x, y):
        return x + y

    closed = jax.make_jaxpr(fn)(jnp.array(2.0), jnp.array(3.0))

    # Fix both inputs
    partial_closed = partial(closed, {0: 7.0, 1: 3.0})

    # The partial function should have no inputs
    assert len(partial_closed.jaxpr.invars) == 0

    # Evaluate the partial function with no inputs
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts))

    # Should be 7.0 + 3.0 = 10.0
    assert jnp.allclose(result[0], 10.0)


def test_partial_with_complex_computation():
    """Partially evaluate a function with more complex operations."""

    def fn(x, y, z):
        a = x + y
        b = a * z
        c = b - x
        return c

    closed = jax.make_jaxpr(fn)(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0))

    # Fix x = 2.0
    partial_closed = partial(closed, {0: 2.0})

    # The partial function should have two inputs (y, z)
    assert len(partial_closed.jaxpr.invars) == 2

    # Evaluate: (2.0 + y) * z - 2.0 with y=1.0, z=3.0 = (2.0 + 1.0) * 3.0 - 2.0 = 7.0
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(1.0), jnp.array(3.0)))
    assert jnp.allclose(result[0], 7.0)


def test_partial_with_constants():
    """Partially evaluate a function that captures constants."""

    CONST = 10.0

    def fn(x, y):
        return x + y + CONST

    closed = jax.make_jaxpr(fn)(jnp.array(1.0), jnp.array(2.0))

    # Fix y = 5.0
    partial_closed = partial(closed, {1: 5.0})

    # The partial function should have one input (x)
    assert len(partial_closed.jaxpr.invars) == 1

    # Evaluate with x = 3.0: 3.0 + 5.0 + 10.0 = 18.0
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(3.0)))
    assert jnp.allclose(result[0], 18.0)


def test_partial_preserves_dtype():
    """Partially evaluate preserves data types through the jaxpr."""

    def fn(x, y):
        return x + y

    # Use integer arrays
    closed = jax.make_jaxpr(fn)(jnp.array(2, dtype=jnp.int32), jnp.array(3, dtype=jnp.int32))

    # Fix first input
    partial_closed = partial(closed, {0: 5})

    # Evaluate with integer input
    result = tuple(eval_jaxpr(partial_closed.jaxpr, partial_closed.consts, jnp.array(3, dtype=jnp.int32)))
    
    # Should still be integer
    assert result[0].dtype == jnp.int32
    assert jnp.allclose(result[0], 8)


def test_partial_input_order_preserved():
    """Verify that unfixed inputs maintain their order in the partial jaxpr."""

    def fn(x, y, z, w):
        return x + y + z + w

    closed = jax.make_jaxpr(fn)(
        jnp.array(1.0), jnp.array(2.0), jnp.array(3.0), jnp.array(4.0)
    )

    # Fix inputs 0 and 2 (x and z)
    partial_closed = partial(closed, {0: 10.0, 2: 20.0})

    # The remaining inputs should be y and w
    assert len(partial_closed.jaxpr.invars) == 2

    # Evaluate: 10.0 + y + 20.0 + w = 10.0 + 2.0 + 20.0 + 4.0 = 36.0
    result = tuple(eval_jaxpr(
        partial_closed.jaxpr, partial_closed.consts, jnp.array(2.0), jnp.array(4.0)
    ))
    assert jnp.allclose(result[0], 36.0)
