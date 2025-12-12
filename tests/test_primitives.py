import pytest
import jax
import jax.numpy as jnp
from jax._src.core import ShapedArray, eval_jaxpr

from taxpr.primitives import iter_tags, tag, dissolve_tags, inject


def _collect_tags(fn, *args):
    closed = jax.make_jaxpr(fn)(*args)
    return list(iter_tags(closed.jaxpr))


def test_iter_tags_single_basic():
    def fn(x):
        return tag(x + 1, op="single", id=1)

    tags = _collect_tags(fn, jnp.array(2.0))
    assert len(tags) == 1
    params, shape = tags[0]
    assert params["op"] == "single"
    assert params["id"] == 1
    assert isinstance(shape, ShapedArray)
    assert shape.shape == ()
    assert shape.dtype == jnp.array(2.0).dtype


def test_iter_tags_nested_multiple():
    def inner(y):
        return tag(y * 2, op="inner", id=2)

    def outer(x):
        first = tag(x, op="outer", id=1)
        second = inner(first)
        return tag(second + 1, op="outer_end", id=3)

    tags = _collect_tags(outer, jnp.array(1.0))
    ids = {p["id"] for p, _ in tags}
    assert ids == {1, 2, 3}
    assert len(tags) == 3


def test_iter_tags_tuple_token_shape():
    def fn(x, y):
        t = (x, y + 1)
        return tag(t, op="tuple", id=5)

    tags = _collect_tags(fn, jnp.ones((2,)), jnp.ones((2,)))
    assert len(tags) == 1
    _, shape = tags[0]
    assert isinstance(shape, tuple) and len(shape) == 2
    assert all(isinstance(s, ShapedArray) for s in shape)
    assert shape[0].shape == (2,) and shape[1].shape == (2,)


def test_iter_tags_jit():
    @jax.jit
    def fn(x):
        return tag(x * 3, op="jit", id=7)

    tags = _collect_tags(fn, jnp.array(1.0))
    ids = {p["id"] for p, _ in tags}
    assert 7 in ids


def test_iter_tags_vmap():
    def fn(x):
        return tag(x * 4, op="vmap", id=8)

    vmapped = jax.vmap(fn)
    tags = _collect_tags(vmapped, jnp.ones((3,)))
    ids = {p["id"] for p, _ in tags}
    assert 8 in ids


def test_iter_tags_custom_jvp_primal():
    @jax.custom_jvp
    def fn(x):
        return tag(x * 2, op="cjvp_primal", id=10)

    @fn.defjvp
    def _fn_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        return tag(x * 2, op="cjvp_primal", id=10), tag(2 * t, op="cjvp_tangent", id=10)

    tags = _collect_tags(fn, jnp.array(2.0))
    ids = {p["id"] for p, _ in tags}
    assert 10 in ids


def test_iter_tags_custom_jvp_rule():
    @jax.custom_jvp
    def fn(x):
        return x * 2

    @fn.defjvp
    def _fn_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        y = tag(x * 2, op="cjvp_rule_primal", id=11)
        ydot = tag(2 * t, op="cjvp_rule_tangent", id=12)
        return y, ydot

    def jvp_call(x):
        y, dydx = jax.jvp(fn, (x,), (jnp.ones_like(x),))
        return y + dydx

    tags = _collect_tags(jvp_call, jnp.array(3.0))
    ids = {p["id"] for p, _ in tags}
    assert {11, 12}.issubset(ids)


def test_iter_tags_no_tags():
    def fn(x):
        return x * 2

    tags = _collect_tags(fn, jnp.array(1.0))
    assert tags == []


# Tests for dissolve_tags

def test_dissolve_tags_single():
    """Test dissolving a single tag from a Jaxpr."""
    def fn(x):
        tagged = tag(x + 1, op="test", id=1)
        return tagged * 2

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    # Before dissolving, should have a tag
    tags_before = list(iter_tags(closed.jaxpr))
    assert len(tags_before) == 1
    
    # After dissolving, should have no tags
    dissolved = dissolve_tags(closed.jaxpr)
    tags_after = list(iter_tags(dissolved))
    assert len(tags_after) == 0
    
    # Result should be the same
    result_before = eval_jaxpr(closed.jaxpr, closed.consts, jnp.array(2.0))[0]
    result_after = eval_jaxpr(dissolved, closed.consts, jnp.array(2.0))[0]
    assert jnp.allclose(result_before, result_after)


def test_dissolve_tags_multiple():
    """Test dissolving multiple tags from a Jaxpr."""
    def fn(x):
        t1 = tag(x + 1, op="add", id=1)
        t2 = tag(t1 * 2, op="mul", id=2)
        return t2 + 3

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    tags_before = list(iter_tags(closed.jaxpr))
    assert len(tags_before) == 2
    
    dissolved = dissolve_tags(closed.jaxpr)
    tags_after = list(iter_tags(dissolved))
    assert len(tags_after) == 0


def test_dissolve_tags_with_predicate():
    """Test dissolving tags based on a predicate."""
    def fn(x):
        t1 = tag(x + 1, op="keep", id=1)
        t2 = tag(t1 * 2, op="remove", id=2)
        return t2 + 3

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    # Dissolve only tags with op="remove"
    def predicate(params, shape):
        return params["op"] == "remove"
    
    dissolved = dissolve_tags(closed.jaxpr, predicate=predicate)
    tags_after = list(iter_tags(dissolved))
    
    # Should have one tag left (the "keep" one)
    assert len(tags_after) == 1
    assert tags_after[0][0]["op"] == "keep"


def test_dissolve_tags_no_tags():
    """Test dissolving when there are no tags."""
    def fn(x):
        return x * 2

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    dissolved = dissolve_tags(closed.jaxpr)
    
    # Should execute without error
    result = eval_jaxpr(dissolved, closed.consts, jnp.array(2.0))[0]
    assert jnp.allclose(result, jnp.array(4.0))


def test_dissolve_tags_jit():
    """Test dissolving tags in a JIT-compiled function."""
    @jax.jit
    def fn(x):
        return tag(x * 2, op="jit", id=7)

    closed = jax.make_jaxpr(fn)(jnp.array(1.0))
    tags_before = list(iter_tags(closed.jaxpr))
    assert len(tags_before) > 0
    
    dissolved = dissolve_tags(closed.jaxpr)
    tags_after = list(iter_tags(dissolved))
    assert len(tags_after) == 0


def test_dissolve_tags_vmap():
    """Test dissolving tags in a vmap'd function."""
    def fn(x):
        return tag(x * 4, op="vmap", id=8)

    vmapped = jax.vmap(fn)
    closed = jax.make_jaxpr(vmapped)(jnp.ones((3,)))
    
    dissolved = dissolve_tags(closed.jaxpr)
    tags_after = list(iter_tags(dissolved))
    assert len(tags_after) == 0


# Tests for inject

def test_inject_basic():
    """Test basic injection of a tag with a simple identity injector."""
    def fn(x):
        return tag(x + 1.0, op="test", id=1) * 2

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    # Simple injector that just returns the value unchanged
    def injector(ctx, token, params):
        ctx = ctx + 1
        return token, ctx

    injected = inject(closed, injector, jnp.array(0))
    result, ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0), jnp.array([0.0, 0.0]))
    
    # Should have called the injector
    assert ctx == 1
    # Result should be the same
    assert jnp.allclose(result[0], jnp.array([2.0, 2.0]))


def test_inject_with_context():
    """Test injection with context mutation."""
    def fn(x):
        t1 = tag(x, op="set", id=1)
        return t1 + 1

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    # Injector that increments context
    def injector(ctx, token, params):
        # token is now in its original structure (scalar in this case)
        ctx = ctx + token
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.array(2.0))
    
    # Context should be updated by adding the token (2.0)
    assert jnp.allclose(final_ctx, jnp.array(2.0))
    # Result should be x + 1 where x = 2.0, so 3.0
    assert jnp.allclose(result, jnp.array(3.0))


def test_inject_multiple_tags():
    """Test injection with multiple tags."""
    def fn(x):
        t1 = tag(x, op="get", id=1)
        t2 = tag(t1 + 1, op="set", id=2)
        return t2

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    injector_calls = []
    def injector(ctx, token, params):
        injector_calls.append(params["op"])
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.array(2.0))
    
    # Should have called injector for both tags
    assert len(injector_calls) == 2
    assert "get" in injector_calls
    assert "set" in injector_calls


def test_inject_no_tags():
    """Test injection when there are no tags."""
    def fn(x):
        return x * 2

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    call_count = [0]
    def injector(ctx, token, params):
        call_count[0] += 1
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.array(2.0))
    
    # Injector should not be called
    assert call_count[0] == 0
    assert jnp.allclose(result, jnp.array(4.0))


def test_inject_jit():
    """Test injection in a JIT-compiled function."""
    @jax.jit
    def fn(x):
        return tag(x * 2, op="jit", id=7)

    closed = jax.make_jaxpr(fn)(jnp.array(1.0))
    
    call_count = [0]
    def injector(ctx, token, params):
        call_count[0] += 1
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.array(1.0))
    
    # Injector should be called
    assert call_count[0] > 0


def test_inject_vmap():
    """Test injection in a vmap'd function."""
    def fn(x):
        return tag(x * 4, op="vmap", id=8)

    vmapped = jax.vmap(fn)
    closed = jax.make_jaxpr(vmapped)(jnp.ones((3,)))
    
    call_count = [0]
    def injector(ctx, token, params):
        call_count[0] += 1
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.ones((3,)))
    
    # Injector should be called
    assert call_count[0] > 0


def test_inject_custom_jvp():
    """Test injection in a custom_jvp function."""
    @jax.custom_jvp
    def fn(x):
        return tag(x * 2, op="cjvp", id=10)

    @fn.defjvp
    def _fn_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        return tag(x * 2, op="cjvp", id=10), tag(2 * t, op="cjvp", id=10)

    closed = jax.make_jaxpr(fn)(jnp.array(2.0))
    
    call_count = [0]
    def injector(ctx, token, params):
        call_count[0] += 1
        return token, ctx

    injected = inject(closed, injector, jnp.array(0.0))
    result, final_ctx = eval_jaxpr(injected.jaxpr, injected.consts, jnp.array(0.0), jnp.array(2.0))
    
    # Injector should be called
    assert call_count[0] > 0
