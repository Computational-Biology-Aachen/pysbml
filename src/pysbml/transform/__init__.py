"""Transform SBML complexity into simpler representation.

Transformations:

- species and parameters
  Use simpler definition of constant parameters and dynamic variables
  SBML parameters **can** change and species **can** be constant
  Regroup these by whether they change.


- When the attribute 'initialAmount' is set, the unit of measurement associated with
    the value of 'initialAmount' is specified by the Species attribute 'substanceUnits'
- When the 'initialConcentration' attribute is set, the unit of measurement
    associated with this concentration value is {unit of amount} divided by
    {unit of size}, where the {unit of amount} is specified by the Species
    'substanceUnits' attribute, and the {unit of size} is specified by the 'units'
    attribute of the Compartment object in which the species is located
- Note that in either case, a unit of amount is involved and determined by
    the 'substanceUnits' attribute
- Note these two attributes alone do not determine the units of the species when
    the species identifier appears in a mathematical expression;
    that aspect is determined by the attribute 'hasOnlySubstanceUnits' discussed below

Additional considerations for interpreting the numerical value of a species
- Species are unique in SBML in that they have a kind of duality:
    a species identifier may stand for either
    - substance amount (meaning, a count of the number of individual entities)
    - a concentration or density (meaning, amount divided by a compartment size).
- When a species definition has a 'hasOnlySubstanceUnits' attribute value of False
    and the size of the compartment in which the species is located changes,
    the default in SBML is to assume that it is the concentration
    that must be updated to account for the size change.
- There is one exception: if the species' quantity is determined by an AssignmentRule,
    RateRule, AlgebraicRule, or an EventAssignment and the species has a
    'hasOnlySubstanceUnits' attribute value of False,
    it means that the concentration is assigned by the rule or event;
    in that case, the amount must be calculated when the compartment size changes
- (Events also require additional care in this situation, because an event with
    multiple assignments could conceivably reassign both a species quantity and a
    compartment size simultaneously.
    Please refer to the SBML specifications for the details.)

- Note that the above only matters if a species has a 'hasOnlySubstanceUnits'
    attribute value of False, meaning that the species identifier refers to a
    concentration wherever the identifier appears in a mathematical formula.
    If instead the attribute's value is True, then the identifier of the species
    always stands for an amount wherever it appears in a mathematical formula or
    is referenced by an SBML construct. In that case, there is never a question about
    whether an assignment or event is meant to affect the amount or concentration:
    it is always the amount.

- A particularly confusing situation can occur when the species has 'constant'
    attribute value of True in combination with a 'hasOnlySubstanceUnits' attribute
    value of False. Suppose this species is given a value for 'initialConcentration'.
    Does a 'constant' value of True mean that the concentration is held constant if
    the compartment size changes? No; it is still the amount that is kept constant
    across a compartment size change. The fact that the species was initialized using
    a concentration value is irrelevant.

Source: https://sbml.org/software/libsbml/5.18.0/docs/formatted/python-api/classlibsbml_1_1_species.html


"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import cast

import sympy

from pysbml.parse import data as pdata
from pysbml.transform.units import CONVERSION, PREFIXES

from . import data
from .mathml2sympy import convert_mathml

LOGGER = logging.getLogger(__name__)


def expr(x: data.Expr | sympy.Basic) -> sympy.Expr:
    return cast(sympy.Expr, x)


@dataclass
class Ctx:
    rxns_by_var: defaultdict[str, set[str]]
    ass_rules_by_var: defaultdict[str, set[str]]


def _to_sympy_types(
    x: str | float | data.Expr,
) -> data.Expr:
    if isinstance(x, str):
        return sympy.Symbol(x)
    if isinstance(x, float | int):
        return sympy.Float(x)
    return x


def _div_expr(
    x: str | float | data.Expr,
    y: str | float | data.Expr,
) -> sympy.Expr:
    return _to_sympy_types(x) / _to_sympy_types(y)  # type: ignore


def _mul_expr(
    x: str | float | data.Expr,
    y: str | float | data.Expr,
) -> sympy.Expr:
    return _to_sympy_types(x) * _to_sympy_types(y)  # type: ignore


def compartment_is_valid(pmodel: pdata.Model, species: pdata.Species) -> bool:
    if (comp := species.compartment) is None:
        return False
    return bool(
        (
            pmodel.compartments[comp].size != 0
            and not math.isnan(pmodel.compartments[comp].size)
        )
        or comp in pmodel.assignment_rules
        or comp in pmodel.initial_assignments
    )


def variable_is_constant(name: str, pmodel: pdata.Model) -> bool:
    var = pmodel.variables[name]
    if var.is_constant:
        return True
    if var.has_boundary_condition:
        return name not in pmodel.rate_rules
    return False


def free_symbols(expr: sympy.Expr) -> list[str]:
    return [i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol)]


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


def convert_functions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, fn in pmodel.functions.items():
        tmodel.functions[name] = convert_mathml(fn.body, fns=tmodel.functions)


def convert_parameters(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, par in pmodel.parameters.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(
                value=sympy.Float(par.value), unit=None
            )
        else:
            tmodel.variables[k] = data.Variable(value=sympy.Float(par.value), unit=None)


def convert_compartments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, par in pmodel.compartments.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(
                value=sympy.Float(par.size), unit=None
            )
        else:
            tmodel.variables[k] = data.Variable(value=sympy.Float(par.size), unit=None)


def convert_rules_and_initial_assignments(
    pmodel: pdata.Model, tmodel: data.Model
) -> None:
    for name, rr in pmodel.rate_rules.items():
        # Rate rules can create variables by SBML spec. Not cool
        if name not in tmodel.variables:
            tmodel.variables[name] = data.Variable(value=sympy.Float(0.0), unit=None)

        tmodel.reactions[f"d{name}"] = data.Reaction(
            expr=convert_mathml(rr.body, fns=tmodel.functions),
            stoichiometry={name: sympy.Float(1.0)},
        )

    for _ in pmodel.algebraic_rules.items():
        msg = "Algebraic rules not yet supported"
        raise NotImplementedError(msg)

    for name, ar in pmodel.assignment_rules.items():
        tmodel.derived[name] = convert_mathml(ar.body, fns=tmodel.functions)

    for name, ia in pmodel.initial_assignments.items():
        tmodel.initial_assignments[name] = convert_mathml(ia.body, fns=tmodel.functions)


def _convert_stoich_tuple(x: tuple[float, str]) -> sympy.Expr:
    factor, name = x
    return sympy.Mul(sympy.Float(factor), sympy.Symbol(name))


def convert_reactions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, rxn in pmodel.reactions.items():
        fn = convert_mathml(rxn.body, fns=tmodel.functions)
        stoichiometry: data.Stoichiometry = {
            k: (_convert_stoich_tuple(v) if isinstance(v, tuple) else sympy.Float(v))
            for k, v in rxn.stoichiometry.items()
        }
        pars_to_replace = {pn: f"{name}_{pn}" for pn in rxn.local_pars}
        fn = expr(fn.subs(pars_to_replace))
        tmodel.reactions[name] = data.Reaction(
            expr=fn,
            stoichiometry=stoichiometry,
        )

        for pn, par in rxn.local_pars.items():
            tmodel.parameters[pars_to_replace[pn]] = data.Parameter(
                value=sympy.Float(par.value), unit=None
            )


def remove_duplicate_entries(tmodel: data.Model) -> None:
    for name in tmodel.derived:
        if name in tmodel.parameters:
            del tmodel.parameters[name]
        elif name in tmodel.variables:
            del tmodel.variables[name]


def _transform_species(
    k: str,
    species: pdata.Species,
    pmodel: pdata.Model,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Separate species into parameters and variables and substitute correct version
    in reactions, rules and initial assignments if necessary.
    """
    if species.conversion_factor is not None:
        raise NotImplementedError

    init = sympy.Float(
        init
        if (init := species.initial_amount) is not None
        or (init := species.initial_concentration) is not None
        else 0.0
    )

    # Now start making case distinctions
    # Easiest is to check first if the compartment is valid
    # If not, our life is significantly easier, because there really are just two choices
    if not compartment_is_valid(pmodel, species=species):
        if variable_is_constant(k, pmodel):
            tmodel.parameters[k] = data.Parameter(value=init, unit=None)
            return

        tmodel.variables[k] = data.Variable(value=init, unit=None)
        return

    # Compartment is valid as in exists and is non-zero / nan
    compartment = cast(str, species.compartment)

    # Now the garbage begins
    # I'm going to do something disgusting by now and write out every case explicitly
    # in a nested way to see the entire decision tree
    # I know this is bad code, I'll refactor it later

    # Let's separate next by is_concentration / is_amount / is_to_be_determined :')
    # Because of course some of them are annotated without either conc or amount

    # We have an amount here
    if species.initial_amount is not None:
        if species.has_only_substance_units:
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: amount | True | True", k)
                tmodel.parameters[k] = data.Parameter(value=init, unit=None)
            else:
                LOGGER.debug("Species %s amount | True | False", k)
                tmodel.variables[k] = data.Variable(value=init, unit=None)
        else:  # noqa: PLR5501
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: amount | False | True", k)

                if k not in pmodel.rate_rules:
                    tmodel.parameters[k] = data.Parameter(value=init, unit=None)
                else:
                    tmodel.variables[k] = data.Variable(value=init, unit=None)

                # We need the concentration of the boundary species in reactions
                k_conc = f"{k}_conc"
                tmodel.derived[k_conc] = _div_expr(k, compartment)

                # Fix reactions
                for rxn_name in ctx.rxns_by_var[k]:
                    rxn = tmodel.reactions[rxn_name]
                    rxn.expr = expr(rxn.expr.subs(k, k_conc))

            else:
                LOGGER.debug("Species %s: amount | False | False", k)
                # This is the default case for most tests
                # We have an amount, that has to be interpreted as a concentration
                # in e.g. reactions, but then the integration has to yield an amount again
                # So divide variable/compartment, but calculate flux*compartment

                tmodel.variables[k] = data.Variable(value=init, unit=None)
                tmodel.derived[k_conc := f"{k}_conc"] = _div_expr(k, compartment)

                # Fix initial assignment rule
                if (ar := tmodel.initial_assignments.get(k)) is not None:
                    LOGGER.debug("Initial assignmet for species %s", k)
                    # If initial assignment updates compartment, use the expression
                    # of the updated compartment
                    tmodel.initial_assignments[k] = _mul_expr(
                        ar,
                        comp
                        if (comp := tmodel.initial_assignments.get(compartment))
                        is not None
                        else compartment,
                    )

                # Fix assignment rules
                # for ar in ctx.ass_rules_by_var[k]:
                #     tmodel.derived[ar] = _div_expr(tmodel.derived[ar], compartment)

                # Fix rate rules
                if (rr := tmodel.reactions.get(f"d{k}")) is not None:
                    rr.stoichiometry = {k: sympy.Symbol(compartment)}
                    # rr.stoichiometry = {k: sympy.Float(1.0)}

                # Fix reaction
                for rxn_name in ctx.rxns_by_var[k]:
                    LOGGER.debug("Fixing reaction %s", rxn_name)
                    rxn = tmodel.reactions[rxn_name]
                    rxn.expr = expr(rxn.expr.subs(k, k_conc))
                    if (s := rxn.stoichiometry.get(k)) is not None:
                        rxn.stoichiometry[k] = _mul_expr(s, compartment)

                    # Since we are inserting a concentration but changing an amount
                    # we need to remove the compartment
                    rxn.expr = expr(
                        rxn.expr.subs(compartment, _div_expr(compartment, compartment))
                    )

    # We have a concentration here
    elif species.initial_concentration is not None:
        # If it IS a concentration but has only substance units
        # is set, we have to multiply it by the compartment initially
        if species.has_only_substance_units:
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: | conc | True | True", k)
                tmodel.variables[k] = data.Variable(value=init, unit=None)
                tmodel.initial_assignments[k] = _mul_expr(init, compartment)

            else:
                LOGGER.debug("Species %s: | conc | True | False", k)
                tmodel.variables[k] = data.Variable(value=init, unit=None)
                tmodel.initial_assignments[k] = _mul_expr(init, compartment)

        else:  # noqa: PLR5501
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: | conc | False | True", k)

                tmodel.variables[k] = data.Variable(value=init, unit=None)
                k_amount = f"{k}_amount"
                tmodel.derived[k_amount] = _div_expr(k, compartment)

                # Fix derived
                # FIXME: this is really inefficient
                for dname, der in tmodel.derived.items():
                    if dname == k_amount:
                        continue
                    tmodel.derived[dname] = expr(der.subs(k, k_amount))

                # Fix rate rules
                if (rr := tmodel.reactions.get(f"d{k}")) is not None:
                    rr.expr = _mul_expr(rr.expr, compartment)

            else:
                LOGGER.debug("Species %s: | conc | False | False", k)

                if species.is_constant:
                    tmodel.parameters[k] = data.Parameter(value=init, unit=None)

                elif not pmodel.compartments[compartment].is_constant:
                    tmodel.variables[k_amount := f"{k}_amount"] = data.Variable(
                        value=init, unit=None
                    )
                    tmodel.derived[k] = _div_expr(k_amount, compartment)
                    tmodel.initial_assignments[k_amount] = _mul_expr(
                        species.initial_concentration, compartment
                    )

                    for rxn_name in ctx.rxns_by_var[k]:
                        rxn = tmodel.reactions[rxn_name]
                        if k in rxn.stoichiometry:
                            rxn.stoichiometry[k_amount] = rxn.stoichiometry.pop(k)

                else:
                    LOGGER.debug("Compartment is constant")
                    # If compartment is constant it's fine to interpret variable as
                    # concentration
                    tmodel.variables[k] = data.Variable(value=init, unit=None)
                    if (
                        comp := tmodel.initial_assignments.get(compartment)
                    ) is not None:
                        tmodel.initial_assignments[k] = _mul_expr(init, comp)

                    # Fix rate rules
                    if (rr := tmodel.derived.get(f"d{k}")) is not None:
                        tmodel.derived[f"d{k}"] = expr(
                            rr.subs(compartment, _div_expr(compartment, compartment))
                        )

                    for rxn_name in ctx.rxns_by_var[k]:
                        rxn = tmodel.reactions[rxn_name]

                        rxn.expr = expr(
                            rxn.expr.subs(
                                compartment, _div_expr(compartment, compartment)
                            )
                        )

    # Now BOTH of them are None, the whackest case of them all. If you think you can
    # figure out if it is a concentration or amount just by looking at species
    # and compartments, boy do I have a surprise for you :)
    else:
        is_concentration = False
        for rxn_name in ctx.rxns_by_var[k]:
            reaction = pmodel.reactions[rxn_name]
            targets = {i.name for i in reaction.args} | set(reaction.stoichiometry)
            for other in targets:
                if (
                    var := pmodel.variables.get(other)
                ) is not None and var.initial_concentration is not None:
                    is_concentration = True
                    break
        # Inject concentration and run the whole thing again to avoid
        # duplicating all those conditions
        if is_concentration:
            pmodel.variables[k].initial_concentration = 0.0
            _transform_species(k, species, pmodel, tmodel, ctx)

        # Fall back to interpretation as amount if no evidence for concentration
        # was found
        else:
            pmodel.variables[k].initial_amount = 0.0
            _transform_species(k, species, pmodel, tmodel, ctx)


def transform_species(pmodel: pdata.Model, tmodel: data.Model, ctx: Ctx) -> None:
    LOGGER.debug("Species name | type | only subs. | boundary cond.")
    for k, var in pmodel.variables.items():
        _transform_species(k, var, pmodel, tmodel, ctx=ctx)


def transform(doc: pdata.Document) -> data.Model:
    for plugin in doc.plugins:
        if plugin.name == "comp":
            msg = "Comp package not yet supported."
            raise NotImplementedError(msg)

    pmodel = doc.model
    if pmodel.conversion_factor is not None:
        msg = "Conversion factors not yet supported"
        raise NotImplementedError(msg)

    ctx = Ctx(rxns_by_var=defaultdict(set), ass_rules_by_var=defaultdict(set))
    for name, rxn in pmodel.reactions.items():
        for arg in rxn.args:
            ctx.rxns_by_var[arg.name].add(name)
        for arg in rxn.stoichiometry:
            ctx.rxns_by_var[arg].add(name)
    for name, rule in pmodel.assignment_rules.items():
        for arg in rule.args:
            ctx.ass_rules_by_var[arg.name].add(name)

    tmodel = data.Model(name=pmodel.name)  # type: ignore
    convert_units(pmodel=pmodel, tmodel=tmodel)
    convert_parameters(pmodel=pmodel, tmodel=tmodel)
    convert_compartments(pmodel=pmodel, tmodel=tmodel)
    convert_constraints(pmodel=pmodel, tmodel=tmodel)
    convert_events(pmodel=pmodel, tmodel=tmodel)
    convert_functions(pmodel=pmodel, tmodel=tmodel)
    convert_rules_and_initial_assignments(pmodel=pmodel, tmodel=tmodel)
    convert_reactions(pmodel=pmodel, tmodel=tmodel)

    # Do the heavy lifting here
    transform_species(pmodel=pmodel, tmodel=tmodel, ctx=ctx)
    remove_duplicate_entries(tmodel=tmodel)
    return tmodel
