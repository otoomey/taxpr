from collections.abc import Mapping
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

def partition_out(jaxpr: core.ClosedJaxpr, outvar_indices: list[int]) -> list[core.ClosedJaxpr]:
    """
    Partition a ClosedJaxpr into multiple ClosedJaxprs, each computing a single outvar.

    Args:
        jaxpr: The ClosedJaxpr to partition.
        outvar_indices: A list of indices of the outvars to partition.
    Returns:
        A list of ClosedJaxpr functions, each computing one of the specified outvars.

    """
    
    # convert jaxpr to rx graph
    dfg, varmap = _dfg(jaxpr.jaxpr)

    functions = []
    for i in outvar_indices:
        outvar = jaxpr.jaxpr.outvars[i]

        # Include the node that produces the outvar as well as its ancestors
        dependencies = set(rx.ancestors(dfg, varmap[outvar]))
        dependencies.add(varmap[outvar])
        
        func_eqns = []

        for eqn in jaxpr.jaxpr.eqns:
            outvar_nodes = [varmap[v] for v in eqn.outvars if isinstance(v, core.Var)]
            if any(node in dependencies for node in outvar_nodes):
                func_eqns.append(eqn)

        func_jaxpr = core.Jaxpr(
            constvars=jaxpr.jaxpr.constvars,
            invars=jaxpr.jaxpr.invars,
            outvars=[outvar],
            eqns=func_eqns,
            effects=jaxpr.jaxpr.effects,
            debug_info=jaxpr.jaxpr.debug_info,
            is_high=jaxpr.jaxpr.is_high,
        )

        closed_jaxpr = core.ClosedJaxpr(func_jaxpr, jaxpr.consts)
        functions.append(closed_jaxpr)

    return functions