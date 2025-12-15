"""
Tests for examples from the usage guide.

These tests verify that all code examples in docs/guide.md work correctly.
"""

import jax
import jax.numpy as jnp
from jax.core import eval_jaxpr

from taxpr.primitives import tag, iter_tags, dissolve_tags, inject


# ============================================================================
# Basic Usage Tests
# ============================================================================


def test_tag_basic_usage():
    """Test basic tagging and that outputs are unchanged."""
    x = jnp.array([1.0, 2.0, 3.0])
    tagged_x = tag(x, debug_name="input", stage="preprocessing")
    assert jnp.allclose(x, tagged_x)


def test_tag_pytree():
    """Test tagging works with PyTree structures."""
    tree = {"weights": jnp.ones((3, 3)), "bias": jnp.zeros(3)}
    tagged_tree = tag(tree, component="model_params")
    
    assert jnp.allclose(tagged_tree["weights"], tree["weights"])
    assert jnp.allclose(tagged_tree["bias"], tree["bias"])


def test_introspection_basic():
    """Test basic jaxpr introspection with tags."""
    def compute(x, y):
        x = tag(x, name="input_x")
        y = tag(y, name="input_y")
        
        z = x + y
        z = tag(z, name="sum")
        
        result = z * jnp.exp(z)
        return tag(result, name="output")
    
    jaxpr = jax.make_jaxpr(compute)(jnp.array(1.0), jnp.array(2.0))
    
    # Collect all tags
    tags = list(iter_tags(jaxpr.jaxpr))
    tag_names = {params["name"] for params, _ in tags}
    
    assert tag_names == {"input_x", "input_y", "sum", "output"}
    assert len(tags) == 4
    
    # Verify shapes
    for params, shape in tags:
        assert shape.shape == ()
        assert shape.dtype == jnp.float32


def test_dissolve_all_tags():
    """Test dissolving all tags from a jaxpr."""
    def compute(x, y):
        x = tag(x, name="input_x")
        y = tag(y, name="input_y")
        z = x + y
        return tag(z, name="sum")
    
    jaxpr = jax.make_jaxpr(compute)(jnp.array(1.0), jnp.array(2.0))
    
    # Verify tags exist before dissolving
    tags_before = list(iter_tags(jaxpr.jaxpr))
    assert len(tags_before) == 3
    
    # Dissolve all tags
    clean_jaxpr = dissolve_tags(jaxpr.jaxpr)
    
    # Verify no tags after dissolving
    tags_after = list(iter_tags(clean_jaxpr))
    assert len(tags_after) == 0


def test_dissolve_with_predicate():
    """Test dissolving tags matching a predicate."""
    def compute(x):
        x = tag(x, name="input")
        y = x * 2
        y = tag(y, name="intermediate")
        z = y + 1
        return tag(z, name="output")
    
    jaxpr = jax.make_jaxpr(compute)(jnp.array(1.0))
    
    # Keep only output tags
    def keep_output_only(params, shape):
        return params["name"] != "output"
    
    filtered_jaxpr = dissolve_tags(jaxpr.jaxpr, predicate=keep_output_only)
    
    # Should have only one tag left
    remaining_tags = list(iter_tags(filtered_jaxpr))
    assert len(remaining_tags) == 1
    assert remaining_tags[0][0]["name"] == "output"


# ============================================================================
# Example 1: State Management Tests
# ============================================================================


def test_state_management_concepts():
    """Test the concepts behind state management with tags.
    
    This demonstrates how tags can be used with inject to manage state.
    """
    
    def simple_fn(x):
        # Tag a "get" operation
        state_val = tag(x, op="get", var_id=0)
        # Tag a "set" operation  
        result = tag(state_val + 1, op="set", var_id=0)
        return result
    
    # Trace the function
    closed_jaxpr = jax.make_jaxpr(simple_fn)(jnp.array(5.0))
    
    # Define an injector that tracks state operations
    def state_injector(state_val, token, params):
        var_id = params["var_id"]
        if params["op"] == "get":
            # Return the current state value
            return state_val, state_val
        elif params["op"] == "set":
            # Update state with new value
            return token, token
        return token, state_val
    
    # Inject the injector with initial state
    initial_state = jnp.array(5.0)
    injected = inject(closed_jaxpr, state_injector, initial_state)
    
    # Execute with proper argument structure
    result, final_state = eval_jaxpr(
        injected.jaxpr,
        injected.consts,
        initial_state,
        jnp.array(5.0)
    )
    
    # The result should be x + 1
    assert jnp.allclose(result, jnp.array(6.0))
    # State should be updated to the new value
    assert jnp.allclose(final_state, jnp.array(6.0))


# ============================================================================
# Example 2: Profiling/Instrumentation Tests
# ============================================================================


def test_computation_profiler_basic():
    """Test basic instrumentation concept with tags.
    
    Demonstrates that tags can be introspected before/after
    to track which computation points are visited.
    """
    
    def neural_net(x):
        # Tag inputs
        x = tag(x, component="input", stage="entry")
        
        # Simulate layer computation
        x = x * 2
        x = tag(x, component="layer1", stage="activation")
        
        # Simulate output
        x = x + 1
        logits = tag(x, component="output", stage="exit")
        return logits
    
    # Trace the function
    closed_jaxpr = jax.make_jaxpr(neural_net)(jnp.array([1.0, 2.0, 3.0]))
    
    # Use iter_tags to find all instrumentation points
    tags = list(iter_tags(closed_jaxpr.jaxpr))
    
    # Verify all instrumentation points are found
    components = {params["component"] for params, _ in tags}
    stages = {params["stage"] for params, _ in tags}
    
    assert "input" in components
    assert "layer1" in components
    assert "output" in components
    
    assert "entry" in stages
    assert "activation" in stages
    assert "exit" in stages
    
    # Verify we have 3 tags
    assert len(tags) == 3


# ============================================================================
# Example 3: Constraint Validation Tests
# ============================================================================


def test_constraint_validator_valid_input():
    """Test constraint tracking with inject.
    
    Demonstrates using inject to track values at constraint points.
    """
    
    def constrained_computation(x):
        """A computation with constraints."""
        # Constraint: x must be non-negative
        x = tag(x, constraint="non-negative", name="input")
        
        # Apply transformations
        y = x * 2
        y = tag(y, constraint="bounded", bounds=(-100, 100), name="doubled")
        
        z = jnp.sqrt(y + 1)
        z = tag(z, constraint="positive", name="sqrt_result")
        
        result = z / (jnp.mean(z) + 0.1)
        return tag(result, constraint="normalized", name="output")
    
    # Trace the function
    closed_jaxpr = jax.make_jaxpr(constrained_computation)(jnp.array([1.0, 2.0, 3.0]))
    
    # Injector that passes through values while accumulating context
    def constraint_injector(count, token, params):
        """Track constraint points via a counter."""
        # Simply increment the count when we hit a constraint
        return token, count + jnp.array(1.0)
    
    # Inject the constraint tracker
    initial_context = jnp.array(0.0)
    injected = inject(closed_jaxpr, constraint_injector, initial_context)
    
    # Execute with valid input
    result, constraint_count = eval_jaxpr(
        injected.jaxpr,
        injected.consts,
        initial_context,
        jnp.array([1.0, 2.0, 3.0])
    )
    
    # Should have hit 4 constraint points
    assert jnp.allclose(constraint_count, jnp.array(4.0))
    assert result.shape == (3,)


def test_constraint_validator_invalid_input():
    """Test constraint tracking with different inputs.
    
    Demonstrates that inject works consistently with different
    input values.
    """
    
    def constrained_computation(x):
        """A computation with constraints."""
        x = tag(x, constraint="non-negative", name="input")
        
        y = x * 2
        y = tag(y, constraint="bounded", bounds=(-100, 100), name="doubled")
        
        z = jnp.sqrt(y + 1)
        z = tag(z, constraint="positive", name="sqrt_result")
        
        result = z / (jnp.mean(z) + 0.1)
        return tag(result, constraint="normalized", name="output")
    
    # Trace the function
    closed_jaxpr = jax.make_jaxpr(constrained_computation)(jnp.array([1.0, 2.0, 3.0]))
    
    # Injector that counts constraint hits
    def constraint_injector(count, token, params):
        """Count each constraint hit."""
        return token, count + jnp.array(1.0)
    
    # Inject the constraint tracker
    initial_context = jnp.array(0.0)
    injected = inject(closed_jaxpr, constraint_injector, initial_context)
    
    # Execute with different input
    result, constraint_count = eval_jaxpr(
        injected.jaxpr,
        injected.consts,
        initial_context,
        jnp.array([2.0, 3.0, 4.0])
    )
    
    # Should have hit 4 constraint points regardless of input
    assert jnp.allclose(constraint_count, jnp.array(4.0))
    assert result.shape == (3,)


# ============================================================================
# Best Practices Tests
# ============================================================================


def test_pytree_with_tags():
    """Test tagging works naturally with PyTree operations."""
    tree = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0, 4.0])}
    
    # Tag each leaf
    tagged_tree = jax.tree.map(
        lambda leaf: tag(leaf, level="input"),
        tree
    )
    
    # Process the tagged tree
    @jax.jit
    def process(tagged_input):
        return jax.tree.map(lambda x: x * 2, tagged_input)
    
    result = process(tagged_tree)
    
    # Verify structure is preserved
    assert "x" in result
    assert "y" in result
    assert jnp.allclose(result["x"], tree["x"] * 2)
    assert jnp.allclose(result["y"], tree["y"] * 2)


def test_tag_with_vmap():
    """Test that tags work with vmap."""
    def compute_with_tag(x):
        x = tag(x, operation="process")
        return x * 2
    
    # Batch the function
    batched = jax.vmap(compute_with_tag)
    
    # Apply to batch
    batch_input = jnp.arange(10, dtype=jnp.float32)
    result = batched(batch_input)
    
    # Verify results
    assert jnp.allclose(result, batch_input * 2)


def test_tag_composition():
    """Test that tags compose well with other operations."""
    def compute_with_tag(x, y):
        x = tag(x, name="input_x")
        y = tag(y, name="input_y")
        return tag(x + y, name="sum")
    
    # Trace and inspect
    jaxpr = jax.make_jaxpr(compute_with_tag)(jnp.array(1.0), jnp.array(2.0))
    
    # Collect tags
    tags = list(iter_tags(jaxpr.jaxpr))
    tag_names = {params["name"] for params, _ in tags}
    
    assert tag_names == {"input_x", "input_y", "sum"}
    assert len(tags) == 3
