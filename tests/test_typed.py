import jax
import jax.numpy as jnp
import functools
from dataclasses import dataclass

from jaxtyping import Array

from taxpr.typed import (
    _to_shape,
    TypedClosedJaxpr,
    tcj_transpose,
    tcj_partition_out,
)
from taxpr.tag import tag


# Custom pytree for testing, similar to the user's X class
@dataclass(init=False)
@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["token"],
    meta_fields=[],
)
class X:
    token: Array
    
    def __init__(self, token: Array):
        self.token = token


def test_to_shape_scalar():
    """Test _to_shape converts scalar arrays to shape/dtype structs."""
    arr = jnp.array(5.0)
    shape_struct = _to_shape(arr)
    
    assert shape_struct.shape == ()
    assert shape_struct.dtype == jnp.float32


def test_to_shape_array():
    """Test _to_shape converts multi-dimensional arrays."""
    arr = jnp.ones((2, 3, 4), dtype=jnp.int32)
    shape_struct = _to_shape(arr)
    
    assert shape_struct.shape == (2, 3, 4)
    assert shape_struct.dtype == jnp.int32


def test_to_shape_pytree():
    """Test _to_shape preserves pytree structure."""
    pytree = {
        'a': jnp.array(1.0),
        'b': jnp.ones((2, 3)),
        'c': (jnp.array(5, dtype=jnp.int32), jnp.array([1, 2, 3], dtype=jnp.int32))
    }
    shape_struct = _to_shape(pytree)
    
    assert shape_struct['a'].shape == ()
    assert shape_struct['a'].dtype == jnp.float32
    assert shape_struct['b'].shape == (2, 3)
    assert shape_struct['b'].dtype == jnp.float32
    assert shape_struct['c'][0].shape == ()
    assert shape_struct['c'][0].dtype == jnp.int32
    assert shape_struct['c'][1].shape == (3,)
    assert shape_struct['c'][1].dtype == jnp.int32


def test_to_shape_multiple_scalars():
    """Test _to_shape with multiple scalar arguments returns tuple."""
    arr1 = jnp.array(1.0)
    arr2 = jnp.array(2, dtype=jnp.int32)
    shape_struct = _to_shape((arr1, arr2))
    
    assert isinstance(shape_struct, tuple)
    assert len(shape_struct) == 2
    assert shape_struct[0].shape == ()
    assert shape_struct[0].dtype == jnp.float32
    assert shape_struct[1].shape == ()
    assert shape_struct[1].dtype == jnp.int32


def test_typed_closed_jaxpr_make_basic():
    """Test TypedClosedJaxpr.make creates a valid typed jaxpr."""
    def fn(x):
        return x + 1.0
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    assert tcj.closed_jaxpr is not None
    # in_shape is a tuple of shape structs (flattened)
    assert len(tcj.in_shape) == 1
    assert tcj.in_shape[0].shape == ()
    assert tcj.in_shape[0].dtype == jnp.float32
    # out_shape is a single shape struct for scalar output
    assert tcj.out_shape.shape == ()
    assert tcj.out_shape.dtype == jnp.float32


def test_typed_closed_jaxpr_make_multiple_inputs():
    """Test TypedClosedJaxpr.make with multiple inputs."""
    def fn(x, y):
        return x + y, x * y
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(2.0), jnp.array(3.0))
    
    # in_shape is a tuple of flattened shapes
    assert len(tcj.in_shape) == 2
    assert tcj.in_shape[0].shape == ()
    assert tcj.in_shape[1].shape == ()
    # out_shape is a tuple of shape structs for tuple output
    assert isinstance(tcj.out_shape, tuple)
    assert len(tcj.out_shape) == 2


def test_typed_closed_jaxpr_eval_basic():
    """Test TypedClosedJaxpr.eval computes correct values."""
    def fn(x):
        return x + 1.0
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    result = tcj.eval(jnp.array(3.0))
    assert jnp.allclose(result, 4.0)


def test_typed_closed_jaxpr_eval_multiple_outputs():
    """Test TypedClosedJaxpr.eval with multiple outputs."""
    def fn(x):
        return x + 1.0, x * 2.0
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    result = tcj.eval(jnp.array(3.0))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert jnp.allclose(result[0], 4.0)
    assert jnp.allclose(result[1], 6.0)


def test_typed_closed_jaxpr_eval_multiple_inputs():
    """Test TypedClosedJaxpr.eval with multiple inputs."""
    def fn(x, y):
        return x + y
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(2.0), jnp.array(3.0))
    
    result = tcj.eval(jnp.array(2.0), jnp.array(3.0))
    assert jnp.allclose(result, 5.0)


def test_typed_closed_jaxpr_eval_pytree():
    """Test TypedClosedJaxpr.eval with pytree inputs."""
    def fn(d):
        return d['x'] + d['y']
    
    maker = TypedClosedJaxpr.make(fn)
    input_dict = {'x': jnp.array(2.0), 'y': jnp.array(3.0)}
    tcj = maker(input_dict)
    
    result = tcj.eval(input_dict)
    assert jnp.allclose(result, 5.0)


def test_typed_closed_jaxpr_remap_inputs_basic():
    """Test TypedClosedJaxpr.remap_inputs remaps input arguments."""
    def original_fn(x, y):
        return x + y
    
    def remap_fn(z):
        # Split z into two arguments
        return (z, z * 2.0)
    
    maker = TypedClosedJaxpr.make(original_fn)
    tcj = maker(jnp.array(2.0), jnp.array(3.0))
    
    remapped = tcj.map_inputs(remap_fn)
    remapped_tcj = remapped(jnp.array(5.0))
    
    # Should have 1 input instead of 2
    assert len(remapped_tcj.in_shape) == 1
    
    # Evaluate: 5.0 + (5.0 * 2.0) = 15.0
    result = remapped_tcj.eval(jnp.array(5.0))
    assert jnp.allclose(result, 15.0)


def test_typed_closed_jaxpr_remap_inputs_with_dict():
    """Test TypedClosedJaxpr.remap_inputs with dict inputs."""
    def original_fn(x, y):
        return x * y
    
    def remap_fn(d):
        return (d['a'], d['b'])
    
    maker = TypedClosedJaxpr.make(original_fn)
    tcj = maker(jnp.array(2.0), jnp.array(3.0))
    
    remapped = tcj.map_inputs(remap_fn)
    remapped_tcj = remapped({'a': jnp.array(4.0), 'b': jnp.array(5.0)})
    
    result = remapped_tcj.eval({'a': jnp.array(4.0), 'b': jnp.array(5.0)})
    assert jnp.allclose(result, 20.0)


def test_typed_closed_jaxpr_remap_outputs_basic():
    """Test TypedClosedJaxpr.remap_outputs remaps output values."""
    def fn(x):
        return x + 1.0
    
    def output_map(x):
        return x * 2.0
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    # remap_outputs returns the remapped TypedClosedJaxpr directly
    remapped_tcj = tcj.map_outputs(output_map)
    
    # Original output: 4.0, after mapping: 8.0
    result = remapped_tcj.eval(jnp.array(3.0))
    assert jnp.allclose(result, 8.0)


def test_typed_closed_jaxpr_remap_outputs_multiple_outputs():
    """Test TypedClosedJaxpr.remap_outputs with multiple outputs."""
    def fn(x):
        return x + 1.0, x * 2.0
    
    def output_map(t):
        a, b = t
        return a + b
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    # remap_outputs returns the remapped TypedClosedJaxpr directly
    remapped_tcj = tcj.map_outputs(output_map)
    
    # Original outputs: (4.0, 6.0), after mapping: 10.0
    result = remapped_tcj.eval(jnp.array(3.0))
    assert jnp.allclose(result, 10.0)


def test_tcj_transpose_basic():
    """Test tcj_transpose transposes a TypedClosedJaxpr."""
    def fn(x):
        return tag(x + 1.0, op="add", id=1)
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    transposed, params = tcj_transpose(tcj)
    
    # Transposed should have same inputs
    assert len(transposed.in_shape) == len(tcj.in_shape)
    # Output should be a list of cotangents
    assert isinstance(transposed.out_shape, list)
    # Params should exist
    assert params is not None


# Tests for tcj_partition_out
def test_tcj_partition_out_single_outputs():
    """Test tcj_partition_out partitions outputs into individual outputs."""
    def fn(x):
        a = x + 1.0
        b = x * 2.0
        return a, b
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    # Partition each output separately
    parts = tcj_partition_out(tcj, [[0], [1]])
    
    assert len(parts) == 2
    
    # First partition computes a = x + 1.0, second output is None
    result0 = parts[0].eval(jnp.array(3.0))
    assert isinstance(result0, tuple) and len(result0) == 2
    assert jnp.allclose(result0[0], 4.0)
    assert result0[1] is None
    
    # Second partition computes b = x * 2.0, first output is None
    result1 = parts[1].eval(jnp.array(3.0))
    assert isinstance(result1, tuple) and len(result1) == 2
    assert result1[0] is None
    assert jnp.allclose(result1[1], 6.0)


def test_tcj_partition_out_single_function_output():
    """Test tcj_partition_out when function returns single non-tuple output."""
    def fn(x):
        return x + 1.0
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    # Partition a single output
    parts = tcj_partition_out(tcj, [[0]])
    
    assert len(parts) == 1
    result = parts[0].eval(jnp.array(3.0))
    # Single output case - result should be the value itself, not wrapped in tuple
    assert jnp.allclose(result, 4.0)

def test_tcj_partition_out_with_custom_pytree():
    """Test tcj_partition_out with custom pytree structures like X."""
    def fn(x):
        a = x + 1.0
        b = x * 2.0
        c = x - 1.0
        return (X(token=a), b), [c, a, b]
    
    maker = TypedClosedJaxpr.make(fn)
    tcj = maker(jnp.array(3.0))
    
    print(f"Original tcj.out_shape: {tcj.out_shape}")
    print(f"  Leaves: {jax.tree.leaves(tcj.out_shape)}")
    
    # Partition into three groups: first has indices 0,1; second has 2,3; third has 4
    parts = tcj_partition_out(tcj, [[0, 1], [2, 3], [4]])

    assert len(parts) == 3
    
    # The return structure is: ((X(token=a), b), [c, a, b])
    # So out_shape is: ((X(token=ShapeDtypeStruct), ShapeDtypeStruct), [ShapeDtypeStruct, ShapeDtypeStruct, ShapeDtypeStruct])
    # Leaves in order: X's token (index 0), b (index 1), c (index 2), a (index 3), b (index 4)
    
    # Check first partition out_shape - should have indices 0,1 which are X's token and b
    out_shape0 = parts[0].out_shape
    print(f"Partition 0 out_shape: {out_shape0}")
    assert isinstance(out_shape0, tuple)
    nested_tuple, list_shapes = out_shape0
    x_shape, b_shape = nested_tuple
    assert isinstance(x_shape, X)
    assert isinstance(x_shape.token, jax.ShapeDtypeStruct)
    assert isinstance(b_shape, jax.ShapeDtypeStruct)
    assert all(s is None for s in list_shapes)
    
    # Check second partition out_shape - should have indices 2,3 which are c and a
    out_shape1 = parts[1].out_shape
    print(f"Partition 1 out_shape: {out_shape1}")
    nested_tuple1, list_shapes1 = out_shape1
    x_shape1, b_shape1 = nested_tuple1
    print(f"  X.token: {x_shape1.token}, type: {type(x_shape1.token)}")
    assert isinstance(x_shape1, X), f"Expected X instance, got {type(x_shape1)}"
    assert x_shape1.token is None, f"Expected None for X.token, got {x_shape1.token}"
    assert b_shape1 is None, f"Expected None for b_shape, got {b_shape1}"
    assert isinstance(list_shapes1, list)
    assert len(list_shapes1) == 3
    assert isinstance(list_shapes1[0], jax.ShapeDtypeStruct), f"Expected ShapeDtypeStruct for c, got {type(list_shapes1[0])}"
    assert isinstance(list_shapes1[1], jax.ShapeDtypeStruct), f"Expected ShapeDtypeStruct for a, got {type(list_shapes1[1])}"
    assert list_shapes1[2] is None, f"Expected None for b in list, got {list_shapes1[2]}"
    
    # Check third partition out_shape - should have index 4 which is b
    out_shape2 = parts[2].out_shape
    print(f"Partition 2 out_shape: {out_shape2}")
    nested_tuple2, list_shapes2 = out_shape2
    x_shape2, b_shape2 = nested_tuple2
    assert isinstance(x_shape2, X)
    assert x_shape2.token is None
    assert b_shape2 is None
    assert all(s is None for s in list_shapes2[:2])
    assert isinstance(list_shapes2[2], jax.ShapeDtypeStruct)
