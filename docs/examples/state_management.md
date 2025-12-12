# Manage State

Tags can be used with `inject()` to thread state through a computation:

```python
import jax
import jax.numpy as jnp
from jax._src.core import eval_jaxpr
from taxpr import tag, inject

def simple_fn(x):
    # Tag a "get" operation to read state
    state_val = tag(x, op="get", var_id=0)
    # Tag a "set" operation to write state
    result = tag(state_val + 1, op="set", var_id=0)
    return result

# Trace the function
closed_jaxpr = jax.make_jaxpr(simple_fn)(jnp.array(5.0))

# Define an injector that handles tagged operations
def state_injector(state_val, token, params):
    if params["op"] == "get":
        # Return current state value, keep state unchanged
        return state_val, state_val
    elif params["op"] == "set":
        # Return the new value, update state
        return token, token
    return token, state_val

# Inject the injector into the traced function
initial_state = jnp.array(5.0)
injected = inject(closed_jaxpr, state_injector, initial_state)

# Execute the injected function
result, final_state = eval_jaxpr(
    injected.jaxpr,
    injected.consts,
    initial_state,
    jnp.array(5.0)
)

print(f"Result: {result}, Final state: {final_state}")
# Output: Result: [6.], Final state: [6.]
```

The key points:
- The injector receives `(context, token, params)` and returns `(new_token, new_context)`
- Context must be JAX-traceable (arrays, not dicts)
- You execute the injected Jaxpr with: `eval_jaxpr(jaxpr, consts, context, *original_args)`
