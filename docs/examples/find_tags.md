
# Finding Tags

Use `iter_tags()` to inspect all the tagged values in a traced computation:

```python
import jax
import jax.numpy as jnp
from taxpr import tag, iter_tags

def neural_net(x, weights):
    # Tag inputs
    x = tag(x, component="input")
    
    # First layer
    z1 = jnp.dot(x, weights[0])
    a1 = tag(jax.nn.relu(z1), component="layer1")
    
    # Second layer
    z2 = jnp.dot(a1, weights[1])
    a2 = tag(jax.nn.relu(z2), component="layer2")
    
    # Output
    logits = tag(jnp.dot(a2, weights[2]), component="output")
    return logits

# Trace the function
x = jnp.ones((32, 784))
weights = [jnp.ones((784, 256)), jnp.ones((256, 256)), jnp.ones((256, 10))]
jaxpr = jax.make_jaxpr(neural_net)(x, weights)

# Find all tags
print("Tags in neural network computation:")
for params, shape in iter_tags(jaxpr.jaxpr):
    component = params["component"]
    print(f"  {component}: shape={shape.shape}, dtype={shape.dtype}")
```

This allows you to:
- Understand which values are tagged at the Jaxpr level
- See shapes of intermediate values after optimization
- Extract metadata about your computation
