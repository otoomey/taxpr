# Track Constraints

Use `inject()` to count constraint points throughout a computation:

```python
import jax
import jax.numpy as jnp
from jax._src.core import eval_jaxpr
from taxpr import tag, inject

def constrained_computation(x):
    """A computation with constraints we want to track."""
    
    x = tag(x, constraint="non-negative", name="input")
    
    y = x * 2
    y = tag(y, constraint="bounded", bounds=(-100, 100), name="doubled")
    
    z = jnp.sqrt(y + 1)
    z = tag(z, constraint="positive", name="sqrt_result")
    
    result = z / (jnp.mean(z) + 0.1)
    return tag(result, constraint="normalized", name="output")

# Trace the function
closed_jaxpr = jax.make_jaxpr(constrained_computation)(jnp.array([1.0, 2.0, 3.0]))

# Injector that counts constraint points
def constraint_counter(count, token, params):
    """Increment counter for each constraint hit."""
    return token, count + jnp.array(1.0)

# Inject the constraint tracker
initial_count = jnp.array(0.0)
injected = inject(closed_jaxpr, constraint_counter, initial_count)

# Execute
result, final_count = eval_jaxpr(
    injected.jaxpr,
    injected.consts,
    initial_count,
    jnp.array([1.0, 2.0, 3.0])
)

print(f"Found {int(final_count)} constraint points")
# Output: Found 4 constraint points
```

This demonstrates using `inject()` to accumulate information as you pass through tagged points in your computation.

## Best Practices

**Using `tag()`**: Tags are transparent to computation - they don't change values, only attach metadata. Always consume the result of `tag()` or it may be optimized away.

**Using `inject()`**:
- Context must be JAX-traceable (arrays, pytrees of arrays - not dicts or Python objects)
- Injector signature: `(context, token, params) -> (new_token, new_context)`
- Execute with: `eval_jaxpr(jaxpr, consts, context, *original_args)`
- Result tuple is: `(original_result, final_context)`

**Tag placement**: Place tags at computation boundaries and key points to track values efficiently.

**With JAX transforms**: Tags work seamlessly with `jit`, `vmap`, `grad`, and other transformations because they're part of the Jaxpr representation.

## Summary

Taxpr provides utilities to work with JAX computations at the Jaxpr level:

- **`tag()`**: Mark values with metadata during tracing
- **`iter_tags()`**: Find all tagged values in a Jaxpr
- **`dissolve_tags()`**: Remove tags selectively
- **`inject()`**: Thread context through tagged points

These primitives enable introspection, instrumentation, and custom control flow while maintaining JAX's purity and composability.
