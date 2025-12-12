import jax
import jax.numpy as jnp
from jax._src.lib import err
from jax._src.tree_util import equality_errors
from jaxtyping import ArrayLike, PyTree


def assert_tree_match(struct1, struct2):
    """Asserts that two JAX tree structures are the same."""
    errors = list(equality_errors(struct1, struct2))
    if len(errors) > 0:
        errors = (str(error) for error in errors)
        raise AssertionError("Tree structures do not match:\n" + "\n".join(errors))


def tree_fill(x: PyTree, v: ArrayLike):
    def _map(a: object):
        if isinstance(a, jax.Array | jax.ShapeDtypeStruct):
            return jnp.full_like(a, v)
        return a

    return jax.tree.map(_map, x)
