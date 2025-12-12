---
icon: lucide/book
---

# Usage Guide

Taxpr provides utilities for manipulating JAX programs by instrumenting them with tags at trace time. This guide covers practical applications through working examples.

## Core Concepts

### What is a Tag?

A tag is a marker you place on arrays during JAX program tracing. Tags carry arbitrary metadata and can be detected, counted, and manipulated from the final compiled JAX program representation (Jaxpr).

### The Tag Workflow

1. **Mark** - Use `tag()` to add metadata to specific arrays during trace time
2. **Extract** - Use `iter_tags()` to find and inspect all tags in a Jaxpr
3. **Transform** - Use `dissolve_tags()` or `inject()` to remove or replace tags with custom logic
4. **Execute** - Evaluate the transformed Jaxpr with modified behavior

## Getting Started: Basic Usage

### Marking Values with Tags

```python
import jax
import jax.numpy as jnp
from taxpr import tag

# Tag a simple value
x = jnp.array([1.0, 2.0, 3.0])
tagged_x = tag(x, debug_name="input", stage="preprocessing")

# Tags work with any PyTree structure
tree = {"weights": jnp.ones((3, 3)), "bias": jnp.zeros(3)}
tagged_tree = tag(tree, component="model_params")

# Tags are transparent - the output value is unchanged
print(jnp.allclose(x, tagged_x))  # True
```

### Inspecting Tags with Introspection

```python
import jax
import jax.numpy as jnp
from taxpr import tag, iter_tags

def compute(x, y):
    x = tag(x, name="input_x")
    y = tag(y, name="input_y")
    
    z = x + y
    z = tag(z, name="sum")
    
    result = z * jnp.exp(z)
    return tag(result, name="output")

# Trace the function to get its Jaxpr
jaxpr = jax.make_jaxpr(compute)(jnp.array(1.0), jnp.array(2.0))

# Inspect all tags
print("Tags found in computation:")
for params, shape in iter_tags(jaxpr.jaxpr):
    print(f"  {params['name']}: shape={shape.shape}, dtype={shape.dtype}")

# Output:
# Tags found in computation:
#   input_x: shape=(), dtype=float32
#   input_y: shape=(), dtype=float32
#   sum: shape=(), dtype=float32
#   output: shape=(), dtype=float32
```

### Removing Tags

Sometimes you want to strip out all tags before execution:

```python
from taxpr import dissolve_tags

jaxpr = jax.make_jaxpr(compute)(jnp.array(1.0), jnp.array(2.0))

# Remove all tags
clean_jaxpr = dissolve_tags(jaxpr.jaxpr)

# Or remove only tags matching a condition
def keep_outputs_only(params, shape):
    return params['name'] != 'output'

filtered_jaxpr = dissolve_tags(jaxpr.jaxpr, predicate=keep_outputs_only)
```