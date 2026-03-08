from collections.abc import Iterator
from typing import Any, Callable
import jax
from jaxtyping import Array, PyTree
from jax.core import ShapedArray, AbstractValue
from jax.interpreters import ad, batching
from jax.extend import core
from jax.core import Atom

from taxpr.dfg import rewrite_invars, rewrite_vars
from taxpr.util import assert_tree_match

tag_p = core.Primitive("tag")


def _rebuild_param_seq(original, new_items: list):
    """Reconstruct a list/tuple/namedtuple preserving the original type."""
    if isinstance(original, list):
        return list(new_items)
    elif type(original) is not tuple:  # NamedTuple or other tuple subclass
        return type(original)(*new_items)
    else:
        return tuple(new_items)


def _tag_p_impl(*token, user_params, structure):
    return tuple(token)


def _tag_p_abstract_eval(*token, user_params, structure):
    return tuple(token)


def _tag_p_jvp(primals, tangents, **params):
    # primals is a tuple when multiple_results=True
    primals_out = tag_p.bind(primals, **params)
    # Ensure tangents is always a tuple
    if not isinstance(tangents, tuple):
        tangents = (tangents,)
    return primals_out, tangents


def _tag_p_batch(batched_args, batch_axes, **params):
    # Bind returns a tuple since multiple_results=True
    result = tag_p.bind(*batched_args, **params)
    # Result is always a tuple when multiple_results=True
    # batch_axes should have the same structure as result
    # Since tag is identity, all outputs have the same batch axis as their corresponding input
    batch_axes_out = batch_axes
    return result, batch_axes_out


tag_p.def_impl(_tag_p_impl)
tag_p.def_abstract_eval(_tag_p_abstract_eval)
tag_p.multiple_results = True
ad.primitive_jvps[tag_p] = _tag_p_jvp
batching.primitive_batchers[tag_p] = _tag_p_batch


def tag[T](token: T, **params) -> T:
    """
    Tag a specific point in a computation with given parameters.

    Note: You must consume the output of this function for the tag to appear
    in the JAXPR. Simply calling this function without using its output
    may lead to the tag being optimized away.

    Args:
        token: An input token representing a point in the computation. The token can be any PyTree of JAX arrays.
        **params: Arbitrary keyword parameters to associate with the tag.
    Returns:
        The unchanged input token tagged with the provided parameters.
    """
    leaves, structure = jax.tree.flatten(token)
    # Convert params to a tuple of items (hashable and immutable)
    # Sort to ensure consistent ordering
    user_params = tuple(sorted(params.items()))
    result = tag_p.bind(*leaves, user_params=user_params, structure=structure)
    return jax.tree.unflatten(structure, result)


def dissolve_tags(
    jaxpr: core.Jaxpr,
    predicate: Callable[[dict[str, Any], PyTree[AbstractValue]], bool] | None = None,
) -> core.Jaxpr:
    """
    Remove all tags from the given jaxpr.

    This will traverse the given jaxpr and all nested jaxprs, removing:

    - all tags if `predicate == None`
    - all tags for which `predicate(..) == True`

    Args:
        jaxpr: A JAXPR potentially containing tagged primitives.
        predicate: An optional function that takes tag parameters and the token shape and returns True
                   if the tag should be dissolved, or False to keep it.
    Returns:
        A new JAXPR with all tags removed.
    """
    varmap = {}
    new_eqns = []

    for eqn in jaxpr.eqns:
        if eqn.primitive is tag_p:
            inshape = jax.tree.unflatten(eqn.params["structure"], eqn.invars)
            if predicate is None or predicate(dict(eqn.params["user_params"]), inshape):
                # Dissolve the tag: map outputs to inputs
                for invar, outvar in zip(eqn.invars, eqn.outvars):
                    varmap[outvar] = varmap.get(invar, invar)
            else:
                # Keep the tag: add it to new_eqns with remapped invars
                invars = []
                for invar in eqn.invars:
                    if isinstance(invar, core.Literal):
                        invars.append(invar)
                    elif invar in varmap:
                        invars.append(varmap[invar])
                    else:
                        invars.append(invar)
                new_eqns.append(eqn.replace(invars=list(invars)))
        else:
            invars = []
            for invar in eqn.invars:
                # Skip Literal objects as they're not hashable
                if isinstance(invar, core.Literal):
                    invars.append(invar)
                elif invar in varmap:
                    invars.append(varmap[invar])
                else:
                    invars.append(invar)
            param_vals = []
            for p in eqn.params.values():
                if isinstance(p, core.Jaxpr):
                    param_vals.append(dissolve_tags(p))
                elif isinstance(p, core.ClosedJaxpr):
                    param_vals.append(
                        core.ClosedJaxpr(dissolve_tags(p.jaxpr), p.consts)
                    )
                elif isinstance(p, tuple):
                    new_items = [
                        (
                            dissolve_tags(x)
                            if isinstance(x, core.Jaxpr)
                            else (
                                core.ClosedJaxpr(dissolve_tags(x.jaxpr), x.consts)
                                if isinstance(x, core.ClosedJaxpr)
                                else x
                            )
                        )
                        for x in p
                    ]
                    param_vals.append(_rebuild_param_seq(p, new_items))
                elif isinstance(p, list):
                    param_vals.append(
                        list(
                            (
                                dissolve_tags(x)
                                if isinstance(x, core.Jaxpr)
                                else (
                                    core.ClosedJaxpr(dissolve_tags(x.jaxpr), x.consts)
                                    if isinstance(x, core.ClosedJaxpr)
                                    else x
                                )
                            )
                            for x in p
                        )
                    )
                else:
                    param_vals.append(p)
            params = dict(zip(eqn.params.keys(), param_vals))
            new_eqns.append(eqn.replace(invars=list(invars), params=params))

    return core.Jaxpr(
        jaxpr.constvars,
        jaxpr.invars,
        jaxpr.outvars,
        new_eqns,
        jaxpr.effects,
        jaxpr.debug_info,
        jaxpr.is_high,
    )


def iter_tags(
    jaxpr: core.Jaxpr,
) -> Iterator[tuple[dict[str, Any], PyTree[AbstractValue]]]:
    """
    Iterate over all tags in the given JAXPR.

    Args:
        jaxpr: A JAXPR potentially containing tagged primitives.

    Yields:
        Tuples of (parameters, ShapeDtypeStruct | None) for each tag.
    """
    for eqn in jaxpr.eqns:
        if eqn.primitive is tag_p:
            params = dict(eqn.params["user_params"])
            invars = eqn.invars
            # Convert Vars to their avals (ShapedArray)
            if len(invars) == 1:
                shape = invars[0].aval
            else:
                shape = tuple(v.aval for v in invars)
            yield params, shape
        else:
            for p in eqn.params.values():
                if isinstance(p, core.Jaxpr):
                    yield from iter_tags(p)
                elif isinstance(p, core.ClosedJaxpr):
                    yield from iter_tags(p.jaxpr)
                elif isinstance(p, (tuple, list)):
                    for x in p:
                        if isinstance(x, core.Jaxpr):
                            yield from iter_tags(x)


def inject[Ctx](
    closed_jaxpr: core.ClosedJaxpr,
    injector: Callable[[Ctx, PyTree[Array], dict[str, Any]], tuple[PyTree[Array], Ctx]],
    ctx: Ctx,
    predicate: Callable[[dict[str, Any], PyTree[AbstractValue]], bool] | None = None,
) -> core.ClosedJaxpr:
    """
    Inject tags into the given JAXPR using the provided injector function.

    Args:
        closed_jaxpr: A closed JAXPR potentially containing tagged primitives.
        injector: A function that takes the context, the token, and tag parameters,
                    and returns a new token and modified context. The injector is converted
                    to a JAXPR internally and inlined at each tag point.
        ctx: The initial context to pass to the injector.
        predicate: An optional function that takes tag parameters and the token shape and returns True
                   if the injector should be applied at that tag, or False to skip injection. If None,
                     the injector is applied at all tags.


    Returns:
        A new closed JAXPR with the injector inlined at each tag point. The modified jaxpr also returns
        the final context as an additional output.
    """
    # Convert context leaves to arrays so they can be traced through make_jaxpr
    ctx = jax.tree.map(lambda x: jax.numpy.array(x) if not isinstance(x, jax.ShapeDtypeStruct) else x, ctx)

    # Flatten context to get individual variables
    ctx_flattened, ctx_struct = jax.tree.flatten(ctx)

    # Create Var objects for context leaves
    ctx_vars: list[core.Var] = []
    for ctx_leaf in ctx_flattened:
        var = core.Var(ShapedArray(ctx_leaf.shape, ctx_leaf.dtype))
        ctx_vars.append(var)

    ctx_len = len(ctx_vars)

    # Helper function to transform a single jaxpr, with a flag to control signature modification
    def transform_jaxpr(jaxpr: core.Jaxpr, modify_signature: bool = True):
        """Transform a jaxpr to inject tags and thread context through it.

        Args:
            jaxpr: The jaxpr to transform
            modify_signature: If True, add context to invars/outvars; if False, keep signature unchanged

        Returns: (new_jaxpr, final_ctx_vars, accumulated_consts)
        """
        new_invars: list[core.Var] = (
            list(ctx_vars) + list(jaxpr.invars)
            if modify_signature
            else list(jaxpr.invars)
        )
        new_consts: list[Any] = []
        new_constvars: list[core.Var] = list(jaxpr.constvars)
        new_eqns = []

        current_ctx_vars = list(ctx_vars)
        varmap = {}

        for eqn in jaxpr.eqns:
            # First recursively process nested jaxprs in parameters
            new_params = {}
            for param_name, param_val in eqn.params.items():
                if isinstance(param_val, core.Jaxpr):
                    # Recursively transform nested jaxpr - don't modify signature for nested jaxprs
                    inner_jaxpr, _, inner_consts = transform_jaxpr(
                        param_val, modify_signature=False
                    )
                    new_params[param_name] = inner_jaxpr
                    new_consts.extend(inner_consts)
                    new_constvars.extend(
                        [v for v in inner_jaxpr.constvars if v not in new_constvars]
                    )
                elif isinstance(param_val, core.ClosedJaxpr):
                    inner_jaxpr, _, inner_consts = transform_jaxpr(
                        param_val.jaxpr, modify_signature=False
                    )
                    new_consts.extend(inner_consts)
                    new_consts.extend(param_val.consts)
                    new_constvars.extend(
                        [v for v in inner_jaxpr.constvars if v not in new_constvars]
                    )
                    new_params[param_name] = core.ClosedJaxpr(
                        inner_jaxpr, param_val.consts
                    )
                elif isinstance(param_val, (tuple, list)):
                    new_param_items = []
                    for item in param_val:
                        if isinstance(item, core.Jaxpr):
                            inner_jaxpr, _, inner_consts = transform_jaxpr(
                                item, modify_signature=False
                            )
                            new_param_items.append(inner_jaxpr)
                            new_consts.extend(inner_consts)
                            new_constvars.extend(
                                [
                                    v
                                    for v in inner_jaxpr.constvars
                                    if v not in new_constvars
                                ]
                            )
                        elif isinstance(item, core.ClosedJaxpr):
                            inner_jaxpr, _, inner_consts = transform_jaxpr(
                                item.jaxpr, modify_signature=False
                            )
                            new_consts.extend(inner_consts)
                            new_consts.extend(item.consts)
                            new_constvars.extend(
                                [
                                    v
                                    for v in inner_jaxpr.constvars
                                    if v not in new_constvars
                                ]
                            )
                            new_param_items.append(
                                core.ClosedJaxpr(inner_jaxpr, item.consts)
                            )
                        else:
                            new_param_items.append(item)
                    new_params[param_name] = _rebuild_param_seq(param_val, new_param_items)
                else:
                    new_params[param_name] = param_val

            eqn = eqn.replace(params=new_params)

            if eqn.primitive is tag_p:
                # Remap invars according to any previous rewrites
                eqn = rewrite_invars(eqn, varmap)
                # Extract abstract values from Var objects - these are the flattened leaves
                avals = [v.aval for v in eqn.invars]
                # Unflatten to get the original structure for type hints
                structure = eqn.params["structure"]
                inshape = jax.tree.unflatten(structure, avals)

                # Parameters for predicate and injector
                params_dict = dict(eqn.params["user_params"])

                # Decide whether to inline the injector for this tag
                should_inject = (
                    predicate is None
                    or predicate(params_dict, inshape)
                )

                if not should_inject:
                    # Keep the tag as-is (with remapped invars)
                    new_eqns.append(eqn)
                else:
                    # Trace through the injector with current ctx and token shape
                    def wrapper(c: Ctx, *token_leaves):
                        token = jax.tree.unflatten(structure, jax.tree.leaves(token_leaves))
                        token, ctx = injector(c, token, params_dict)
                        assert_tree_match(jax.tree.structure(token), structure)
                        assert_tree_match(jax.tree.structure(ctx), ctx_struct)
                        return token, ctx

                    inner_closed_jaxpr = jax.make_jaxpr(wrapper)(ctx, inshape)
                    inner_jaxpr: core.Jaxpr = inner_closed_jaxpr.jaxpr

                    new_consts.extend(inner_closed_jaxpr.consts)
                    new_constvars.extend(list(inner_jaxpr.constvars))

                    inner_varmap = {}
                    for ctx_var, inner_ctx_var in zip(
                        current_ctx_vars, inner_jaxpr.invars[:ctx_len]
                    ):
                        inner_varmap[inner_ctx_var] = ctx_var
                    for invar, inner_invar in zip(eqn.invars, inner_jaxpr.invars[ctx_len:]):
                        inner_varmap[inner_invar] = invar

                    inner_jaxpr = rewrite_vars(inner_jaxpr, inner_varmap)

                    for outvar, inner_outvar in zip(
                        eqn.outvars, inner_jaxpr.outvars[:-ctx_len]
                    ):
                        varmap[outvar] = inner_outvar
                    current_ctx_vars = inner_jaxpr.outvars[-ctx_len:]

                    for inner_eqn in inner_jaxpr.eqns:
                        new_eqns.append(inner_eqn)
            else:
                eqn = rewrite_invars(eqn, varmap)
                new_eqns.append(eqn)

        new_outvars: list[Atom] = []
        for outvar in jaxpr.outvars:
            if isinstance(outvar, core.Var) and outvar in varmap:
                new_outvars.append(varmap[outvar])
            else:
                new_outvars.append(outvar)

        if modify_signature:
            for var in current_ctx_vars:
                new_outvars.append(var)

        new_jaxpr = core.Jaxpr(
            new_constvars,
            new_invars,
            new_outvars,
            new_eqns,
            jaxpr.effects,
            jaxpr.debug_info,
            jaxpr.is_high,
        )
        return new_jaxpr, current_ctx_vars, new_consts

    # Transform the outer jaxpr with signature modification
    jaxpr = closed_jaxpr.jaxpr
    new_jaxpr, _, accumulated_consts = transform_jaxpr(jaxpr, modify_signature=True)

    # Combine original constants with accumulated constants
    new_consts = list(closed_jaxpr.consts) + accumulated_consts

    return core.ClosedJaxpr(new_jaxpr, new_consts)

def transpose(
    jaxpr: core.ClosedJaxpr,
):
    """
    Transposes a closed JAXPR such that it returns the outputs of each tag rather than its original outputs.

    Args:
        jaxpr: A closed JAXPR potentially containing tagged primitives.
    Returns:
        A new closed JAXPR that returns the outputs of each tag.
        A list of tag output parameters.
        A list of tag output structures.
    """
    new_outvars: list[core.Var] = []
    outvar_params: list[dict[str, Any]] = []
    outvar_structs: list[PyTree[AbstractValue]] = []
    # Helper: transform a nested core.Jaxpr so that it returns any tag outputs
    def _transform_core_jaxpr(cj: core.Jaxpr):
        new_constvars: list[core.Var] = list(cj.constvars)
        new_consts: list[Any] = []

        # Recursively transform nested jaxprs in params
        new_eqns = []
        collected_inner_params: list[dict[str, Any]] = []
        collected_inner_structs: list[PyTree[AbstractValue]] = []
        for eqn in cj.eqns:
            new_params = {}
            for name, val in eqn.params.items():
                if isinstance(val, core.Jaxpr):
                    inner_jaxpr, inner_constvars, inner_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(val)
                    # accumulate nested inner tag metadata
                    if inner_outvar_params:
                        collected_inner_params.extend(inner_outvar_params)
                        collected_inner_structs.extend(inner_outvar_structs)
                    new_consts.extend(inner_consts)
                    # extend constvars with any new ones
                    for v in inner_constvars:
                        if v not in new_constvars:
                            new_constvars.append(v)
                    new_params[name] = inner_jaxpr
                elif isinstance(val, core.ClosedJaxpr):
                    inner_jaxpr, inner_constvars, inner_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(val.jaxpr)
                    if inner_outvar_params:
                        collected_inner_params.extend(inner_outvar_params)
                        collected_inner_structs.extend(inner_outvar_structs)
                    new_consts.extend(inner_consts)
                    for v in inner_constvars:
                        if v not in new_constvars:
                            new_constvars.append(v)
                    new_params[name] = core.ClosedJaxpr(inner_jaxpr, val.consts)
                elif isinstance(val, (tuple, list)):
                    items = []
                    for item in val:
                        if isinstance(item, core.Jaxpr):
                            inner_jaxpr, inner_constvars, inner_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(item)
                            if inner_outvar_params:
                                collected_inner_params.extend(inner_outvar_params)
                                collected_inner_structs.extend(inner_outvar_structs)
                            new_consts.extend(inner_consts)
                            for v in inner_constvars:
                                if v not in new_constvars:
                                    new_constvars.append(v)
                            items.append(inner_jaxpr)
                        elif isinstance(item, core.ClosedJaxpr):
                            inner_jaxpr, inner_constvars, inner_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(item.jaxpr)
                            if inner_outvar_params:
                                collected_inner_params.extend(inner_outvar_params)
                                collected_inner_structs.extend(inner_outvar_structs)
                            new_consts.extend(inner_consts)
                            for v in inner_constvars:
                                if v not in new_constvars:
                                    new_constvars.append(v)
                            items.append(core.ClosedJaxpr(inner_jaxpr, item.consts))
                        else:
                            items.append(item)
                    new_params[name] = _rebuild_param_seq(val, items)
                else:
                    new_params[name] = val

            eqn = eqn.replace(params=new_params)
            new_eqns.append(eqn)

        # Collect tag outvars present in this core jaxpr
        tag_outvars: list[core.Var] = []
        for eqn in new_eqns:
            if eqn.primitive is tag_p:
                tag_outvars.extend(eqn.outvars)

        # Append those tag outvars to the jaxpr outputs (so callers can observe them)
        if tag_outvars:
            new_outvars = list(cj.outvars) + list(tag_outvars)
        else:
            new_outvars = list(cj.outvars)

        new_cj = core.Jaxpr(
            new_constvars,
            list(cj.invars),
            new_outvars,
            new_eqns,
            cj.effects,
            cj.debug_info,
            cj.is_high,
        )

        # Build outvar params and structures for tags in this inner jaxpr
        inner_outvar_params: list[dict[str, Any]] = []
        inner_outvar_structs: list[PyTree[AbstractValue]] = []
        for eqn in new_eqns:
            if eqn.primitive is tag_p:
                for _ in eqn.outvars:
                    inner_outvar_params.append(dict(eqn.params["user_params"]))
                    inner_outvar_structs.append(eqn.params["structure"])

        # include any collected inner params/structs from recursive calls
        if collected_inner_params:
            inner_outvar_params = collected_inner_params + inner_outvar_params
            inner_outvar_structs = collected_inner_structs + inner_outvar_structs

        return new_cj, new_constvars, new_consts, inner_outvar_params, inner_outvar_structs

    # We'll build a new set of constvars & consts if nested jaxprs generate new ones
    accumulated_constvars: list[core.Var] = list(jaxpr.jaxpr.constvars)
    accumulated_consts: list[Any] = []

    new_eqns = []
    for eqn in jaxpr.jaxpr.eqns:
        if eqn.primitive is tag_p:
            new_outvars.extend(eqn.outvars)
            for _ in eqn.outvars:
                outvar_params.append(dict(eqn.params["user_params"]))
                outvar_structs.append(eqn.params["structure"])
            new_eqns.append(eqn)
        else:
            # Transform nested jaxprs in params so they return tag outputs
            new_params = {}
            extra_outvars_for_eqn: list[core.Var] = []
            for name, val in eqn.params.items():
                if isinstance(val, core.Jaxpr):
                    new_cj, new_constvars, new_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(val)
                    new_params[name] = new_cj
                    # accumulate constvars/consts
                    for v in new_constvars:
                        if v not in accumulated_constvars:
                            accumulated_constvars.append(v)
                    accumulated_consts.extend(new_consts)
                    # add inner outvar params and create outer vars to capture the extra outputs
                    if inner_outvar_params:
                        outvar_params.extend(inner_outvar_params)
                        outvar_structs.extend(inner_outvar_structs)
                        # compute how many extra outvars were appended
                        extra = len(new_cj.outvars) - len(val.outvars)
                        for v in new_cj.outvars[-extra:]:
                            extra_outvars_for_eqn.append(core.Var(v.aval))
                elif isinstance(val, core.ClosedJaxpr):
                    new_cj, new_constvars, new_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(val.jaxpr)
                    new_params[name] = core.ClosedJaxpr(new_cj, val.consts)
                    for v in new_constvars:
                        if v not in accumulated_constvars:
                            accumulated_constvars.append(v)
                    accumulated_consts.extend(new_consts)
                    if inner_outvar_params:
                        outvar_params.extend(inner_outvar_params)
                        outvar_structs.extend(inner_outvar_structs)
                        extra = len(new_cj.outvars) - len(val.jaxpr.outvars)
                        for v in new_cj.outvars[-extra:]:
                            extra_outvars_for_eqn.append(core.Var(v.aval))
                elif isinstance(val, (tuple, list)):
                    new_items = []
                    for item in val:
                        if isinstance(item, core.Jaxpr):
                            new_cj, new_constvars, new_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(item)
                            new_items.append(new_cj)
                            for v in new_constvars:
                                if v not in accumulated_constvars:
                                    accumulated_constvars.append(v)
                            accumulated_consts.extend(new_consts)
                            if inner_outvar_params:
                                outvar_params.extend(inner_outvar_params)
                                outvar_structs.extend(inner_outvar_structs)
                                extra = len(new_cj.outvars) - len(item.outvars)
                                for v in new_cj.outvars[-extra:]:
                                    extra_outvars_for_eqn.append(core.Var(v.aval))
                        elif isinstance(item, core.ClosedJaxpr):
                            new_cj, new_constvars, new_consts, inner_outvar_params, inner_outvar_structs = _transform_core_jaxpr(item.jaxpr)
                            new_items.append(core.ClosedJaxpr(new_cj, item.consts))
                            for v in new_constvars:
                                if v not in accumulated_constvars:
                                    accumulated_constvars.append(v)
                            accumulated_consts.extend(new_consts)
                            if inner_outvar_params:
                                outvar_params.extend(inner_outvar_params)
                                outvar_structs.extend(inner_outvar_structs)
                                extra = len(new_cj.outvars) - len(item.jaxpr.outvars)
                                for v in new_cj.outvars[-extra:]:
                                    extra_outvars_for_eqn.append(core.Var(v.aval))
                        else:
                            new_items.append(item)
                    new_params[name] = _rebuild_param_seq(val, new_items)
                else:
                    new_params[name] = val

            # If this eqn's nested jaxprs produced extra outputs, extend eqn.outvars so
            # the outer jaxpr captures them, and add them to new_outvars to be returned.
            if extra_outvars_for_eqn:
                new_outs = list(eqn.outvars) + extra_outvars_for_eqn
                new_eqns.append(eqn.replace(params=new_params, outvars=new_outs))
                new_outvars.extend(extra_outvars_for_eqn)
            else:
                new_eqns.append(eqn.replace(params=new_params))

    # Build the final new jaxpr combining accumulated constvars and transformed eqns
    final_constvars = accumulated_constvars
    final_consts = list(jaxpr.consts) + accumulated_consts

    new_jaxpr = core.Jaxpr(
        final_constvars,
        jaxpr.jaxpr.invars,
        new_outvars,
        new_eqns,
        jaxpr.jaxpr.effects,
        jaxpr.jaxpr.debug_info,
        jaxpr.jaxpr.is_high,
    )

    return core.ClosedJaxpr(new_jaxpr, final_consts), outvar_params, outvar_structs