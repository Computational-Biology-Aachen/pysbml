"""Transform SBML complexity into simpler representation.

Transformations:

- species and parameters
  Use simpler definition of constant parameters and dynamic variables
  SBML parameters **can** change and species **can** be constant
  Regroup these by whether they change.


"""

import logging
from functools import reduce
from operator import mul
from typing import cast

import sympy

from pysbml.parse import data as pdata
from pysbml.transform.units import CONVERSION, PREFIXES

from . import data
from .mathml2sympy import convert_mathml

LOGGER = logging.getLogger(__name__)


def convert_units(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Replace SBML units with sympy ones."""

    for name, unit in pmodel.atomic_units.items():
        if (mapped := CONVERSION.get(name)) is None:
            LOGGER.warning("Could not map unit %s", name)
            continue
        if (prefix := PREFIXES.get(unit.exponent - 1)) is not None:
            mapped *= prefix
        tmodel.units[name] = mapped

    for name, cunit in pmodel.composite_units.items():
        tmodel.units[name] = reduce(
            mul,
            [tmodel.units.get(k, 1) for k in cunit.units],
        )


def convert_constraints(
    pmodel: pdata.Model,
    tmodel: data.Model,  # noqa: ARG001
) -> None:
    for _ in pmodel.constraints.items():
        msg = "Constraint handling not yet supported"
        raise NotImplementedError(msg)


def convert_events(
    pmodel: pdata.Model,
    tmodel: data.Model,  # noqa: ARG001
) -> None:
    for _ in pmodel.events.items():
        msg = "Event handling not yet supported"
        raise NotImplementedError(msg)


def convert_parameters(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, par in pmodel.parameters.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(value=par.value, unit=None)
        else:
            tmodel.variables[k] = data.Variable(initial_value=par.value, unit=None)


def convert_species(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, var in pmodel.variables.items():
        if var.conversion_factor is not None:
            raise NotImplementedError

        if var.is_constant:
            tmodel.parameters[k] = data.Parameter(value=var.initial_amount, unit=None)
        else:
            tmodel.variables[k] = data.Variable(
                initial_value=var.initial_amount, unit=None
            )


def convert_compartments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, par in pmodel.compartments.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(value=par.size, unit=None)
        else:
            tmodel.variables[k] = data.Variable(initial_value=par.size, unit=None)


def convert_functions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, fn in pmodel.functions.items():
        tmodel.functions[name] = data.Function(
            body=convert_mathml(fn.body, fns=tmodel.functions),
            args=[i.name for i in fn.args],
        )


def convert_rules(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, rr in pmodel.rate_rules.items():
        tmodel.reactions[f"d{name}"] = data.Reaction(
            fn=convert_mathml(rr.body, fns=tmodel.functions),
            args=[i.name for i in rr.args],
            stoichiometry={name: 1},
        )
        # if name not in tmodel.variables:
        #     tmodel.variables[name] = data.Variable(initial_value=0.0, unit=None)
    for _ in pmodel.algebraic_rules.items():
        msg = "Algebraic rules not yet supported"
        raise NotImplementedError(msg)

    for name, ar in pmodel.assignment_rules.items():
        tmodel.derived[name] = data.Derived(
            fn=convert_mathml(ar.body, fns=tmodel.functions),
            args=[i.name for i in ar.args],
        )


def _convert_substance_amount_to_concentration(
    fn: sympy.Expr, args: list[str], pmodel: pdata.Model
) -> tuple[sympy.Expr, list[str]]:
    replacements = {}
    for arg in args:
        # the parsed species part is important to not
        # introduce conversion on things that aren't species
        if (species := pmodel.variables.get(arg)) is None:
            continue
        if species.has_only_substance_units:
            continue
        if (compartment := species.compartment) is None:
            continue
        if pmodel.compartments[compartment].dimensions == 0:
            continue

        if not species.is_concentration:
            old = sympy.Symbol(arg)
            replacements[old] = old / sympy.Symbol(compartment)  # type: ignore
    fn = cast(sympy.Expr, fn.subs(replacements))
    return fn, [i.name for i in cast(list, fn.free_symbols)]


def convert_reactions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, rxn in pmodel.reactions.items():
        fn = convert_mathml(rxn.body, fns=tmodel.functions)
        fn, args = _convert_substance_amount_to_concentration(
            fn=fn,
            args=[i.name for i in rxn.args],
            pmodel=pmodel,
        )
        stoichiometry = rxn.stoichiometry
        pars_to_replace = {pn: f"{name}_{pn}" for pn in rxn.local_pars}
        fn = cast(sympy.Expr, fn.subs(pars_to_replace))

        tmodel.reactions[name] = data.Reaction(
            fn=fn,
            args=[pars_to_replace.get(i, i) for i in args],
            stoichiometry=stoichiometry,
        )

        for pn, par in rxn.local_pars.items():
            tmodel.parameters[pars_to_replace[pn]] = data.Parameter(
                value=par.value, unit=None
            )


def _convert_initial_assignment_concentration(
    fn: sympy.Expr, args: list[str], pmodel: pdata.Model
) -> tuple[sympy.Expr, list[str]]:
    replacements = {}
    for arg in args:
        # the parsed species part is important to not
        # introduce conversion on things that aren't species
        if (species := pmodel.variables.get(arg)) is None:
            continue

        compartment = species.compartment
        if compartment is not None and (
            not species.is_concentration
            or (species.has_only_substance_units and species.is_concentration)
        ):
            size = 1 if (c := pmodel.compartments.get(compartment)) is None else c.size
            if size == 0:
                continue

            old = sympy.Symbol(arg)
            replacements[old] = old * sympy.Symbol(compartment)  # type: ignore
    fn = cast(sympy.Expr, fn.subs(replacements))
    return fn, [i.name for i in cast(list, fn.free_symbols)]


def convert_initial_assignments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, ia in pmodel.initial_assignments.items():
        if (pp := pmodel.parameters.get(name)) is not None and pp.is_constant:
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue
        if (pv := pmodel.variables.get(name)) is not None and pv.is_constant:
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue
        if (el := pmodel.compartments.get(name)) is not None and el.is_constant:
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue

        # Otherwise it's a normal variable
        fn, args = _convert_initial_assignment_concentration(
            fn=convert_mathml(ia.body, fns=tmodel.functions),
            args=[i.name for i in ia.args],
            pmodel=pmodel,
        )
        derived = data.Derived(fn=fn, args=args)
        variable = tmodel.variables.get(name)
        if variable is None:
            tmodel.variables[name] = data.Variable(initial_value=0.0, unit=None)

        tmodel.initial_assignments[name] = derived


def remove_duplicate_entries(tmodel: data.Model) -> None:
    for name in tmodel.derived:
        if name in tmodel.parameters:
            del tmodel.parameters[name]
        elif name in tmodel.variables:
            del tmodel.variables[name]


# def apply_functions(tmodel: data.Model) -> None:


def transform(pmodel: pdata.Model) -> data.Model:
    tmodel = data.Model(name=pmodel.name)
    convert_units(pmodel=pmodel, tmodel=tmodel)
    convert_parameters(pmodel=pmodel, tmodel=tmodel)
    convert_species(pmodel=pmodel, tmodel=tmodel)
    convert_compartments(pmodel=pmodel, tmodel=tmodel)

    convert_constraints(pmodel=pmodel, tmodel=tmodel)
    convert_events(pmodel=pmodel, tmodel=tmodel)
    convert_functions(pmodel=pmodel, tmodel=tmodel)
    convert_rules(pmodel=pmodel, tmodel=tmodel)
    convert_reactions(pmodel=pmodel, tmodel=tmodel)
    convert_initial_assignments(pmodel=pmodel, tmodel=tmodel)
    remove_duplicate_entries(tmodel=tmodel)
    return tmodel
