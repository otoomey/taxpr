from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax.core import AbstractValue
from jax.extend import core
from jaxtyping import Array, PyTree

from taxpr.dfg import partition_out
from taxpr.tag import inject, transpose


def _to_shape(pytree: PyTree) -> PyTree:
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), pytree
    )


@dataclass(frozen=True)
class TypedClosedJaxpr[*T, R]:
    in_shape: PyTree
    closed_jaxpr: core.ClosedJaxpr
    out_shape: PyTree
    static_argnums: int | Iterable[int] = ()

    @staticmethod
    def make[*T_new, R_new](
        fn: Callable[[*T_new], R_new], static_argnums: int | Iterable[int] = ()
    ) -> Callable[..., "TypedClosedJaxpr[*T_new, R_new]"]:
        func = jax.make_jaxpr(fn, static_argnums=static_argnums, return_shape=True)

        def wrapper(*args: *T_new, **kwargs):
            closed_jaxpr, out_shape = func(*args, **kwargs)
            return TypedClosedJaxpr[*T_new, R_new](
                _to_shape(args), closed_jaxpr, out_shape, static_argnums
            )

        return wrapper

    def eval(
        self,
        *args: *T,
    ) -> R:
        out = core.jaxpr_as_fun(self.closed_jaxpr)(*jax.tree.leaves(args))
        return jax.tree.unflatten(jax.tree.structure(self.out_shape), out)

    def map_inputs[*K](
        self, map: Callable[[*K], tuple[*T]], static_argnums: int | Iterable[int] = ()
    ) -> "Callable[..., TypedClosedJaxpr[*K, R]]":
        def new_fn(*args: *K) -> R:
            remapped_args = map(*args)
            return self.eval(*remapped_args)

        return TypedClosedJaxpr.make(new_fn, static_argnums=static_argnums)

    def map_outputs[S](self, map: Callable[[R], S]) -> "TypedClosedJaxpr[*T, S]":
        def new_fn(*args: *T) -> S:
            out = self.eval(*args)
            return map(out)

        return TypedClosedJaxpr.make(new_fn, self.static_argnums)(*self.in_shape)

    def inject[Ctx](
        self,
        injector: Callable[[Ctx, Any, dict[str, Any]], tuple[Any, Ctx]],
        ctx: Ctx,
        predicate: (
            Callable[[dict[str, Any], PyTree[AbstractValue]], bool] | None
        ) = None,
    ) -> "TypedClosedJaxpr[tuple[Ctx, *T], tuple[R, Ctx]]":
        closed_jaxpr = inject(self.closed_jaxpr, injector, ctx, predicate)
        in_shape = _to_shape((ctx, *self.in_shape))
        out_shape = _to_shape((self.out_shape, ctx))
        return TypedClosedJaxpr(
            in_shape, closed_jaxpr, out_shape, static_argnums=self.static_argnums
        )


def tcj_transpose[*T, R](
    tcj: TypedClosedJaxpr[*T, R],
) -> tuple[TypedClosedJaxpr[*T, list[PyTree[Array]]], list[dict[str, Any]]]:
    new_closed_jaxpr, outvar_params, outvar_structs = transpose(tcj.closed_jaxpr)

    return (
        TypedClosedJaxpr[*T, list[PyTree[Array]]](
            in_shape=tcj.in_shape,
            closed_jaxpr=new_closed_jaxpr,
            out_shape=list(outvar_structs),
            static_argnums=tcj.static_argnums,
        ),
        outvar_params,
    )


def tcj_partition_out[*T, *R](
    tcj: TypedClosedJaxpr[*T, tuple[*R]], outvar_indices: list[list[int]]
) -> list[TypedClosedJaxpr[*T, Any]]:
    parts = partition_out(tcj.closed_jaxpr, outvar_indices)

    # Get leaves and structure of original output shape
    out_shape_leaves = jax.tree.leaves(tcj.out_shape)

    typed_parts = []
    for i, group_indices in enumerate(outvar_indices):
        # Convert output indices to a set for fast lookup
        group_set = set(group_indices)

        # Create masked leaves: keep leaves whose indices are in the group, set others to None
        masked_leaves = [
            out_shape_leaves[j] if j in group_set else None
            for j in range(len(out_shape_leaves))
        ]

        # Reconstruct using tree.map on the structure to properly handle custom pytrees
        # by mapping None values through the tree structure
        out_shape_leaves_iter = iter(masked_leaves)

        def get_next_leaf(x):
            return next(out_shape_leaves_iter)

        out_shape = jax.tree.map(get_next_leaf, tcj.out_shape)

        typed_parts.append(
            TypedClosedJaxpr[*T, Any](
                in_shape=tcj.in_shape,
                closed_jaxpr=parts[i],
                out_shape=out_shape,
                static_argnums=tcj.static_argnums,
            )
        )

    return typed_parts
