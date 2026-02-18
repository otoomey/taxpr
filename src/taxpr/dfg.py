from collections.abc import Mapping, Set
from typing import Any
from jax.core import Atom
from jax.extend import core

import rustworkx as rx


def rewrite_invars(eqn: core.JaxprEqn, varmap: Mapping[core.Var, Atom]):
    """Rewrite the invars of an equation according to a variable mapping."""
    invars = []
    for invar in eqn.invars:
        if isinstance(invar, core.Literal):
            invars.append(invar)
        elif invar in varmap:
            invars.append(varmap[invar])
        else:
            invars.append(invar)
    return eqn.replace(invars=invars)


def rewrite_outvars(eqn: core.JaxprEqn, varmap: Mapping[core.Var, Atom]):
    """Rewrite the outvars of an equation according to a variable mapping."""
    outvars = []
    for outvar in eqn.outvars:
        if isinstance(outvar, core.Literal):
            outvars.append(outvar)
        elif outvar in varmap:
            outvars.append(varmap[outvar])
        else:
            outvars.append(outvar)
    return eqn.replace(outvars=outvars)


def inline_jaxpr(eqn: core.JaxprEqn):
    """Inline a jaxpr contained in an equation."""
    assert "jaxpr" in eqn.params, "Equation does not contain a jaxpr to inline."
    inner_jaxpr = eqn.params["jaxpr"]

    varmap = {}

    # Map the invars
    for invar, inner_invar in zip(eqn.invars, inner_jaxpr.invars):
        varmap[inner_invar] = invar

    # Inline the equations
    for inner_eqn in inner_jaxpr.eqns:
        inner_eqn_invars = []
        for invar in inner_eqn.invars:
            if isinstance(invar, core.Literal):
                inner_eqn_invars.append(invar)
            elif invar in varmap:
                inner_eqn_invars.append(varmap[invar])
            else:
                inner_eqn_invars.append(invar)


def rewrite_vars(jaxpr: core.Jaxpr, varmap: Mapping[core.Var, core.Var]):
    """Rewrite the invars and outvars of a jaxpr contained in an equation."""
    new_eqns = []
    for eqn in jaxpr.eqns:
        new_eqn = rewrite_invars(eqn, varmap)
        new_eqn = rewrite_outvars(new_eqn, varmap)
        new_eqns.append(new_eqn)
    new_invars = [varmap.get(var, var) for var in jaxpr.invars]
    new_outvars = [
        varmap.get(var, var) if isinstance(var, core.Var) else var
        for var in jaxpr.outvars
    ]
    return core.Jaxpr(
        jaxpr.constvars,
        new_invars,
        new_outvars,
        new_eqns,
        jaxpr.effects,
        jaxpr.debug_info,
        jaxpr.is_high,
    )

def _dfg(jaxpr: core.Jaxpr):
    """
    Convert a JAX jaxpr into a rustwork DataFlowGraph.

    Args:
        jaxpr: The ClosedJaxpr to convert.

    Returns:
        A rustwork DataFlowGraph representing the jaxpr.
    """
    dfg = rx.PyDiGraph()

    var_nodes: dict[core.Var, int] = {}

    # Add input nodes
    for invar in jaxpr.invars:
        node = dfg.add_node(str(invar))
        var_nodes[invar] = node

    # Add constant nodes
    for constvar in jaxpr.constvars:
        node = dfg.add_node(str(constvar))
        var_nodes[constvar] = node

    # Add operation nodes
    for eqn in jaxpr.eqns:
        input_nodes = [var_nodes[invar] for invar in eqn.invars if isinstance(invar, core.Var)]
        op_node = dfg.add_node(eqn.primitive.name)
        _ = dfg.add_edges_from_no_data((inp, op_node) for inp in input_nodes)
        for outvar in eqn.outvars:
            var_nodes[outvar] = op_node

    return dfg, var_nodes

def partition_out(jaxpr: core.ClosedJaxpr, outvar_indices: list[list[int]]) -> list[core.ClosedJaxpr]:
    """
    Partition a ClosedJaxpr into multiple ClosedJaxprs, each computing a single outvar.

    Args:
        jaxpr: The ClosedJaxpr to partition.
        outvar_indices: The list of outvar indices each partition should compute.
    Returns:
        A list of ClosedJaxprs, each computing the specified outvars.

    """
    
    # convert jaxpr to rx graph
    dfg, varmap = _dfg(jaxpr.jaxpr)

    functions = []
    for group in outvar_indices:
        outvars = [jaxpr.jaxpr.outvars[i] for i in group]

        # Include the node that produces the outvar as well as its ancestors
        dependencies = set()
        for outvar in outvars:
            # Only process Vars; Literals don't have nodes in the DFG
            if isinstance(outvar, core.Var):
                var_dep = set(rx.ancestors(dfg, varmap[outvar]))
                # include the node that directly produces the outvar
                var_dep.add(varmap[outvar])
                dependencies.update(var_dep)
        
        func_eqns = []

        for eqn in jaxpr.jaxpr.eqns:
            outvar_nodes = [varmap[v] for v in eqn.outvars if isinstance(v, core.Var)]
            if any(node in dependencies for node in outvar_nodes):
                func_eqns.append(eqn)

        func_jaxpr = core.Jaxpr(
            constvars=jaxpr.jaxpr.constvars,
            invars=jaxpr.jaxpr.invars,
            outvars=list(outvars),
            eqns=func_eqns,
            effects=jaxpr.jaxpr.effects,
            debug_info=jaxpr.jaxpr.debug_info,
            is_high=jaxpr.jaxpr.is_high,
        )

        closed_jaxpr = core.ClosedJaxpr(func_jaxpr, jaxpr.consts)
        functions.append(closed_jaxpr)

    return functions

def partial(jaxpr: core.ClosedJaxpr, /, args: dict[int, Any]):
    """
    Partially evaluate a ClosedJaxpr by fixing certain input arguments.

    Similar to functools.partial, but for JAX ClosedJaxprs.

    Args:
        jaxpr: The ClosedJaxpr to partially evaluate.
        args: A dictionary mapping input argument indices to their fixed values.

    Returns:
        A new ClosedJaxpr with the specified inputs fixed. The number of inputs
        in the returned jaxpr is the original number of inputs minus the number of fixed inputs.
    """
    varmap = {}
    new_invars = []

    for i, invar in enumerate(jaxpr.jaxpr.invars):
        if i in args:
            varmap[invar] = core.Literal(args[i], invar.aval)
        else:
            new_invars.append(invar)

    new_jaxpr = rewrite_vars(jaxpr.jaxpr, varmap)
    new_jaxpr = new_jaxpr.replace(invars=new_invars)

    return core.ClosedJaxpr(new_jaxpr, jaxpr.consts)

def strip_outputs(jaxpr: core.ClosedJaxpr, /, indices: Set[int]):
    """
    Strip specified outputs from a ClosedJaxpr.

    Args:
        jaxpr: The ClosedJaxpr to strip outputs from.
        indices: A set of output indices to remove.
    Returns:
        A new ClosedJaxpr with the specified outputs removed.
    """
    new_outvars = [
        var for i, var in enumerate(jaxpr.jaxpr.outvars) if i not in indices
    ]

    new_jaxpr = jaxpr.jaxpr.replace(outvars=new_outvars)

    return core.ClosedJaxpr(new_jaxpr, jaxpr.consts)