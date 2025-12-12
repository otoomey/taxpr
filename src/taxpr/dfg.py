from collections.abc import Mapping
from jax.core import Atom
from jax.extend import core


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
    assert 'jaxpr' in eqn.params, "Equation does not contain a jaxpr to inline."
    inner_jaxpr = eqn.params['jaxpr']
    ctx_len = eqn.params.get('ctx_len', 0)

    varmap = {}
    new_eqns = []

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
    new_outvars = [varmap.get(var, var) if isinstance(var, core.Var) else var for var in jaxpr.outvars]
    return core.Jaxpr(jaxpr.constvars, new_invars, new_outvars, new_eqns, jaxpr.effects, jaxpr.debug_info, jaxpr.is_high)


