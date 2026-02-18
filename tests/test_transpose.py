import jax
import jax.numpy as jnp
from taxpr.tag import tag, transpose


def test_transpose_basic_tag():
    x = jnp.ones((3, 4))

    def f(x):
        return tag(x, name="t1")

    closed = jax.make_jaxpr(f)(x)
    new_closed, outvar_params, outvar_structs = transpose(closed)

    assert hasattr(new_closed, "jaxpr")
    assert isinstance(outvar_params, list)
    assert len(outvar_params) == 1
    params = outvar_params[0]
    struct = outvar_structs[0]
    assert params["name"] == "t1"
    assert struct == jax.tree.structure(x)
    assert len(new_closed.jaxpr.outvars) == 1


def test_transpose_multiple_tags():
    x = jnp.ones((2,))
    y = jnp.ones((2,))

    def f(x, y):
        a = tag(x, t="a")
        b = tag(y, t="b")
        return a, b

    closed = jax.make_jaxpr(f)(x, y)
    new_closed, outvar_params, outvar_structs = transpose(closed)

    assert len(outvar_params) == 2
    names = {p["t"] for p in outvar_params}
    assert names == {"a", "b"}
    assert len(new_closed.jaxpr.outvars) == 2


def test_transpose_no_tags():
    x = jnp.ones((2,))

    def f(x):
        return x + 1

    closed = jax.make_jaxpr(f)(x)
    new_closed, outvar_params, outvar_structs = transpose(closed)

    assert outvar_params == []
    assert outvar_structs == []
    assert len(new_closed.jaxpr.outvars) == 0


def test_transpose_with_jit_nested():
    x = jnp.ones((2,))

    def inner(x):
        return tag(x, nested="jit")

    def f(x):
        return jax.jit(inner)(x)

    closed = jax.make_jaxpr(f)(x)
    new_closed, outvar_params, outvar_structs = transpose(closed)

    # Should find the nested tag from the jitted inner function
    assert any(p.get("nested") == "jit" for p in outvar_params)
    # The new jaxpr should expose the nested tag outputs at top-level
    assert len(new_closed.jaxpr.outvars) >= 1


def test_transpose_with_custom_jvp_nested():
    x = jnp.ones((2,))

    @jax.custom_jvp
    def g(x):
        return tag(x, nested="cjvp")

    @g.defjvp
    def g_jvp(primals, tangents):
        return g(primals), tangents

    def f(x):
        return g(x)

    closed = jax.make_jaxpr(f)(x)
    new_closed, outvar_params, outvar_structs = transpose(closed)

    # Should find the nested tag from the custom_jvp'd function
    assert any(p.get("nested") == "cjvp" for p in outvar_params)
    # The new jaxpr should expose the nested tag outputs at top-level
    assert len(new_closed.jaxpr.outvars) >= 1


def test_transpose_deeply_nested_triggers_unpack_error():
    x = jnp.ones((2,))

    def deep(x):
        return tag(x, lvl="deep")

    def mid(x):
        return jax.jit(deep)(x)

    def outer(x):
        return jax.jit(mid)(x)

    closed = jax.make_jaxpr(outer)(x)

    # After fixing _transform_core_jaxpr, transpose should succeed and expose the deep tag
    new_closed, outvar_params, outvar_structs = transpose(closed)
    assert any(p.get("lvl") == "deep" for p in outvar_params)
    assert any(s == jax.tree.structure(x) for s in outvar_structs)
