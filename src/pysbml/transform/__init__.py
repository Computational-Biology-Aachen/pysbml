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

import dataclasses
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import cast

import sympy

from pysbml.parse import data as pdata
from pysbml.parse.mathml import Base as MathMLBase
from pysbml.parse.mathml import Symbol as MathMLSymbol
from pysbml.transform.units import CONVERSION, PREFIXES

from . import data
from .mathml2sympy import SBMLDelay, convert_mathml

LOGGER = logging.getLogger(__name__)


def expr(x: data.Expr | sympy.Basic) -> sympy.Expr:
    """Cast a sympy expression to sympy.Expr for type narrowing."""
    return cast(sympy.Expr, x)


@dataclass
class Ctx:
    """Context tracking which reactions and assignment rules reference each variable."""

    rxns_by_var: defaultdict[str, set[str]]
    ass_rules_by_var: defaultdict[str, set[str]]


def _to_sympy_types(
    x: str | float | data.Expr | sympy.Basic,
) -> data.Expr:
    if isinstance(x, str):
        return sympy.Symbol(x)
    if isinstance(x, float | int):
        return sympy.Float(x)
    return x  # type: ignore


def _div_expr(
    x: str | float | data.Expr | sympy.Basic,
    y: str | float | data.Expr | sympy.Basic,
) -> sympy.Expr:
    return _to_sympy_types(x) / _to_sympy_types(y)  # type: ignore


def _mul_expr(
    x: str | float | data.Expr | sympy.Basic,
    y: str | float | data.Expr | sympy.Basic,
) -> sympy.Expr:
    return _to_sympy_types(x) * _to_sympy_types(y)  # type: ignore


def _mathml_symbol_names(node: MathMLBase) -> set[str]:
    """Recursively collect all Symbol names from a MathML AST node."""
    if isinstance(node, MathMLSymbol):
        return {node.name}
    result: set[str] = set()
    for f in dataclasses.fields(node):  # type: ignore[arg-type]
        val = getattr(node, f.name)
        if isinstance(val, MathMLBase):
            result |= _mathml_symbol_names(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, MathMLBase):
                    result |= _mathml_symbol_names(item)
    return result


def _compartment_in_algebraic_rules(comp: str, pmodel: pdata.Model) -> bool:
    """Return True if comp appears as a free symbol in any algebraic rule."""
    return any(
        comp in _mathml_symbol_names(rule.body)
        for rule in pmodel.algebraic_rules.values()
    )


def compartment_is_valid(pmodel: pdata.Model, species: pdata.Species) -> bool:
    """Return True if the species' compartment exists and has a non-zero, non-nan size."""
    if (comp := species.compartment) is None:
        return False
    return bool(
        (
            pmodel.compartments[comp].size != 0
            and not math.isnan(pmodel.compartments[comp].size)
        )
        or comp in pmodel.assignment_rules
        or comp in pmodel.initial_assignments
        or _compartment_in_algebraic_rules(comp, pmodel)
    )


def variable_is_constant(name: str, pmodel: pdata.Model) -> bool:
    """Return True if the named species should be treated as a constant parameter."""
    var = pmodel.variables[name]
    if var.is_constant:
        return True
    if var.has_boundary_condition:
        return name not in pmodel.rate_rules
    return False


def free_symbols(expr: sympy.Expr) -> list[str]:
    """Return list of free symbol names in a sympy expression."""
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
    """Skip constraints — not modelled in the ODE system."""
    for name in pmodel.constraints:
        LOGGER.warning("Constraint %s ignored (not supported)", name)


def convert_events(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Convert SBML events to sympy trigger and assignment expressions."""
    for name, pevent in pmodel.events.items():
        if pevent.trigger is None or pevent.trigger.math is None:
            LOGGER.warning("Event %s has no trigger math, skipping", name)
            continue

        trigger = convert_mathml(pevent.trigger.math, fns=tmodel.functions)

        assignments: dict[str, sympy.Expr] = {}
        for assignment in pevent.assignments:
            if assignment.math is None:
                LOGGER.warning(
                    "Event %s assignment for %s has no math, skipping",
                    name,
                    assignment.variable,
                )
                continue
            assignments[assignment.variable] = convert_mathml(
                assignment.math, fns=tmodel.functions
            )

        delay = (
            convert_mathml(pevent.delay.math, fns=tmodel.functions)
            if pevent.delay is not None and pevent.delay.math is not None
            else None
        )
        priority = (
            convert_mathml(pevent.priority.math, fns=tmodel.functions)
            if pevent.priority is not None and pevent.priority.math is not None
            else None
        )

        tmodel.events[name] = data.Event(
            trigger=trigger,
            assignments=assignments,
            initial_value=pevent.trigger.initial_value,
            persistent=pevent.trigger.persistent,
            delay=delay,
            priority=priority,
            use_values_from_trigger_time=pevent.use_values_from_trigger_time,
        )


def convert_functions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Convert SBML function definitions to sympy expressions in the transformed model."""
    for name, fn in pmodel.functions.items():
        tmodel.functions[name] = convert_mathml(fn.body, fns=tmodel.functions)


def convert_parameters(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Split SBML parameters into constant parameters and dynamic variables."""
    for k, par in pmodel.parameters.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(
                value=sympy.Float(par.value), unit=None
            )
        else:
            tmodel.variables[k] = data.Variable(value=sympy.Float(par.value), unit=None)


def convert_compartments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Split SBML compartments into constant parameters and dynamic variables."""
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
    """Convert SBML rules and initial assignments to reactions and derived expressions."""
    for name, rr in pmodel.rate_rules.items():
        # Rate rules can create variables by SBML spec. Not cool
        if name not in tmodel.variables:
            tmodel.variables[name] = data.Variable(value=sympy.Float(0.0), unit=None)

        tmodel.reactions[f"d{name}"] = data.Reaction(
            expr=convert_mathml(rr.body, fns=tmodel.functions),
            stoichiometry={name: sympy.Float(1.0)},
        )

    # Algebraic rules are processed after transform_species in convert_algebraic_rules

    for name, ar in pmodel.assignment_rules.items():
        tmodel.derived[name] = convert_mathml(ar.body, fns=tmodel.functions)

    for name, ia in pmodel.initial_assignments.items():
        tmodel.initial_assignments[name] = convert_mathml(ia.body, fns=tmodel.functions)


def _convert_stoich_tuple(x: tuple[float, str]) -> sympy.Expr:
    """Convert a (factor, species_ref) stoichiometry tuple to a sympy expression."""
    factor, name = x
    return sympy.Mul(sympy.Float(factor), sympy.Symbol(name))


def convert_reactions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Convert SBML reactions to sympy kinetic-law expressions with resolved stoichiometry."""
    for name, rxn in pmodel.reactions.items():
        fn = convert_mathml(rxn.body, fns=tmodel.functions)
        stoichiometry: data.Stoichiometry = {}
        for k, v in rxn.stoichiometry.items():
            if isinstance(v, list):
                for tpl in v:
                    stoichiometry[k] = stoichiometry.get(
                        k, sympy.Float(0.0)
                    ) + _convert_stoich_tuple(tpl)  # type: ignore
            else:
                stoichiometry[k] = stoichiometry.get(k, sympy.Float(0.0)) + sympy.Float(
                    v
                )  # type: ignore

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
    """Remove parameters and variables that are superseded by derived expressions."""
    for name in tmodel.derived:
        if name in tmodel.parameters:
            del tmodel.parameters[name]
        elif name in tmodel.variables:
            del tmodel.variables[name]


def _handle_amount(
    k: str,
    compartment: str,
    init: sympy.Float,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle a species given in amount units (default case).

    SBML reactions are written in a way that contains the compartment, e.g.

    | Reaction         | k1 Unit   | Unit      | Test examples |
    | ---------------- | --------- | --------- |               |
    | C * S1 * k1      | 1/s       | L * mol/s | 1             |
    | C * S1 * S2 * k1 | L/(mol*s) | L^2*mol/s | 52, 76        |
    | C * S1^2 * k1    | L/(mol*s) | L^2*mol/s | 52, 76        |


    This means the stoichiometries need to be adjusted accordingly
        1 substrate  => rxn / C
        2 substrates => rxn / C^2
    and so on

    *Counting* this out is in principle doable, but annoying.

    So let's transform the amounts into concentrations instead. This gives a somewhat
    sub-optimal representation because we need to do more work per integration step, but
    is much easier to handle.

    Steps:
      - derive concentration as amount / compartment
      - substitute amount with concentration in reactions & rules
      - remove compartment from reactions & rules
      - multiply stoichiometry by compartment to get amount again

    Initial assignment
      - initial assignments are apparently also given in units of concentration so we
        need to multiply by the compartment here
      - if the compartment also has an initial assignment, use the updated value
        (this should be handled by sorting the dependencies in our case)

    Assignment rules (676, 677, 678)
      - assigns an amount and is given an amount, nothing to fix here

    Rate rules
      - they are always given in units of concentration, so we need to multiply
        by the concentration to get an amount

    """
    tmodel.variables[k] = data.Variable(value=init, unit=None)
    tmodel.derived[k_conc := f"{k}_conc"] = _div_expr(k, compartment)

    # Fix initial assignment rule
    if (ar := tmodel.initial_assignments.get(k)) is not None:
        tmodel.initial_assignments[k] = _mul_expr(ar, compartment)

    # Fix assignment rules
    # Nothing to do here :)

    # Fix rate rule
    if (rr := tmodel.reactions.get(f"d{k}")) is not None:
        rr.stoichiometry = {k: sympy.Symbol(compartment)}

    # Fix reactions
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]
        rxn.expr = expr(
            rxn.expr.subs(
                {
                    compartment: 1,
                    k: k_conc,
                }
            )
        )
        if (s := rxn.stoichiometry.get(k)) is not None:
            rxn.stoichiometry[k] = _mul_expr(s, compartment)

    # Fix derived (assignment) rules: k appears as concentration in math
    k_sym = sympy.Symbol(k)
    k_conc_sym = sympy.Symbol(k_conc)
    for dname in ctx.ass_rules_by_var[k]:
        if dname != k:
            tmodel.derived[dname] = expr(tmodel.derived[dname].subs(k_sym, k_conc_sym))

    # Fix events: hasOnlySubstanceUnits=False means k in expressions = concentration
    k_conc_expr = _div_expr(k, compartment)
    for ev in tmodel.events.values():
        if ev.trigger is not None:
            ev.trigger = expr(ev.trigger.subs(k_sym, k_conc_expr))
        new_assignments = {}
        for var, assign_expr in ev.assignments.items():
            new_expr = expr(assign_expr.subs(k_sym, k_conc_expr))
            if var == k:
                # Assignment sets concentration → convert to amount
                new_expr = _mul_expr(new_expr, compartment)
            new_assignments[var] = new_expr
        ev.assignments = new_assignments


def _handle_amount_boundary(
    k: str,
    compartment: str,
    init: sympy.Float,
    pmodel: pdata.Model,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle amount that has a boundary condition.

    This means, per the spec (4.6.6) that the species is on the boundary of the reaction
    system, and its amount is not determined by the reactions.

    This does **not** mean, that it cannot be changed, because it **can** be changed by
    rate rules.

    """
    if k not in pmodel.rate_rules:
        _handle_constant_variable(k, init=init, tmodel=tmodel, ctx=ctx)
    else:
        tmodel.variables[k] = data.Variable(value=init, unit=None)
    tmodel.derived[k_conc := f"{k}_conc"] = _div_expr(k, compartment)

    # Fix initial assignment rule
    if (ar := tmodel.initial_assignments.get(k)) is not None:
        tmodel.initial_assignments[k] = _mul_expr(ar, compartment)

    # Fix derived (assignment) rules: k appears as concentration in math
    k_sym = sympy.Symbol(k)
    k_conc_sym = sympy.Symbol(k_conc)
    for dname in ctx.ass_rules_by_var[k]:
        if dname != k:
            tmodel.derived[dname] = expr(tmodel.derived[dname].subs(k_sym, k_conc_sym))

    # Fix rate rule
    if (rr := tmodel.reactions.get(f"d{k}")) is not None:
        rr.stoichiometry = {k: sympy.Symbol(compartment)}

    # Fix reactions
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]

        rate = expr(rxn.expr.subs(compartment, 1))

        # FIXME: Test 1122 requires me to divide the concentration by the compartment
        # to run through. That's certainly false. What the hell is this?
        rxn.expr = expr(rate.subs(k, _div_expr(k_conc, compartment)))

        # Boundary condition means it cannot be part of the reactions system, so we
        # don't need to worry about the stoichiometry

    # Fix events: hasOnlySubstanceUnits=False means k in expressions = concentration
    k_conc_expr = _div_expr(k, compartment)
    for ev in tmodel.events.values():
        if ev.trigger is not None:
            ev.trigger = expr(ev.trigger.subs(k_sym, k_conc_expr))
        new_assignments = {}
        for var, assign_expr in ev.assignments.items():
            new_expr = expr(assign_expr.subs(k_sym, k_conc_expr))
            if var == k:
                # Assignment sets concentration → convert to amount
                new_expr = _mul_expr(new_expr, compartment)
            new_assignments[var] = new_expr
        ev.assignments = new_assignments


def _handle_amount_has_substance_units(
    k: str,
    init: sympy.Float,
    tmodel: data.Model,
) -> None:
    """Handle amount with has_substance_units=True.

    According to the spec (4.6.5) the `hasOnlySubstanceUnits` allows choosing the
    meaning intended for a species' identifier when the identifier appears in
    mathematical expressions or as the subject of SBML rules or assignments.
    If the value is false the unit of measurement is a concentration or density, else
    it is always interpreted as having an amount.

    The only way that leads to legal expressions is when they don't contain a compartment
    So we can just essentially ignore everything

    """
    tmodel.variables[k] = data.Variable(value=init, unit=None)

    # Fix initial assignment
    # Nothing to do here :)

    # Fix other assignment rules
    # Nothing to do here :)

    # Fix rate rule
    # Nothing to do here :)

    # Fix reactions
    # Nothing to do here :)


def _handle_amount_boundary_has_substance_units(
    k: str,
    compartment: str,
    init: sympy.Float,
    pmodel: pdata.Model,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle amount with boundary=True and has_substance_units=True.

    A boundary condition means, per the spec (4.6.6), that the species is on the
    boundary of the reaction system, and its amount is not determined by the reactions.

    According to the spec (4.6.5) the `hasOnlySubstanceUnits` allows choosing the
    meaning intended for a species' identifier when the identifier appears in
    mathematical expressions or as the subject of SBML rules or assignments.
    If the value is true the unit of measurement is always interpreted as an amount.

    In test 1123 we have
        S1: only substance units = False
        S3: only substance units = True

        rxn J0 = S3 / 10 with stoichiometry S1: -1.0

    Since by our logic S1 will give itself the stoichiometry S1: -comp, we need to ignore
    the hasOnlySubstanceUnits logic and insert the concentration into the reaction

    FIXME: why does this happen with the boundary, but not without?
    """
    if k not in pmodel.rate_rules:
        _handle_constant_variable(k, init=init, tmodel=tmodel, ctx=ctx)
    else:
        tmodel.variables[k] = data.Variable(value=init, unit=None)

    # We need the concentration of the boundary species in reactions
    k_conc = f"{k}_conc"
    tmodel.derived[k_conc] = _div_expr(k, compartment)

    # Fix reactions
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]
        rxn.expr = expr(rxn.expr.subs(k, k_conc))


def _handle_constant_variable(
    k: str, init: sympy.Float, tmodel: data.Model, ctx: Ctx
) -> None:
    tmodel.parameters[k] = data.Parameter(value=init, unit=None)

    for name in ctx.rxns_by_var[k]:
        tmodel.reactions[name].stoichiometry.pop(k, None)


def _handle_conc(
    k: str,
    compartment: str,
    init: sympy.Float,
    pmodel: pdata.Model,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle species given as a concentration.

    By default this is straightforward, as species are interpreted as concentrations in
    all expressions (except if hasOnlySubstanceUnits=True).

    There is a slight complication though if a compartment is changing. In that case it
    is easier to also create the amount.
    """
    if pmodel.compartments[compartment].is_constant:
        tmodel.variables[k] = data.Variable(value=init, unit=None)

        # Fix initial assignment rule
        if (comp := tmodel.initial_assignments.get(compartment)) is not None:
            tmodel.initial_assignments[k] = _mul_expr(init, comp)

        # Fix assignment rule
        # Nothing to do here :)

        # Fix rate rule
        # Nothing to do here :)

        # Fix reactions
        for rxn_name in ctx.rxns_by_var[k]:
            rxn = tmodel.reactions[rxn_name]
            rxn.expr = expr(rxn.expr.subs(compartment, 1))
    else:
        tmodel.variables[k_amount := f"{k}_amount"] = data.Variable(
            value=init, unit=None
        )
        tmodel.derived[k] = _div_expr(k_amount, compartment)

        # Fix initial assignment rule
        if (ar := tmodel.initial_assignments.get(k)) is not None:
            tmodel.initial_assignments[k_amount] = _mul_expr(ar, compartment)
        else:
            tmodel.initial_assignments[k_amount] = _mul_expr(init, compartment)

        # Fix assignment rules
        # Nothing to do here :)

        # Fix rate rule
        # Nothing to do here :)

        # Fix reactions
        for rxn_name in ctx.rxns_by_var[k]:
            rxn = tmodel.reactions[rxn_name]
            if k in rxn.stoichiometry:
                rxn.stoichiometry[k_amount] = rxn.stoichiometry.pop(k)


def _handle_conc_boundary(
    k: str,
    compartment: str,
    init: sympy.Float,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle species given as a concentration with a boundary condition."""
    tmodel.variables[k_conc := f"{k}_conc"] = data.Variable(value=init, unit=None)
    tmodel.derived[k] = _mul_expr(k_conc, compartment)

    # Fix initial assignment rule
    if (ia := tmodel.initial_assignments.pop(k, None)) is not None:
        tmodel.initial_assignments[k_conc] = ia

    # Fix assignment rules
    for dname in ctx.ass_rules_by_var[k]:
        if dname == k:
            tmodel.derived[k_conc] = expr(tmodel.derived.pop(dname).subs(k, k_conc))
        else:
            tmodel.derived[dname] = expr(tmodel.derived[dname].subs(k, k_conc))

    # Fix rate rule
    if (rr := tmodel.reactions.pop(f"d{k}", None)) is not None:
        tmodel.reactions[f"d{k_conc}"] = rr

        if rr.stoichiometry.get(k) is not None:
            rr.stoichiometry[k_conc] = rr.stoichiometry.pop(k)

    # Fix reactions
    # Nothing to do here, boundary species cannot have reactions :)

    # Fix events: k in expressions = concentration = k_conc
    k_sym = sympy.Symbol(k)
    k_conc_sym = sympy.Symbol(k_conc)
    comp_sym = sympy.Symbol(compartment)
    for ev in tmodel.events.values():
        if ev.trigger is not None:
            ev.trigger = expr(ev.trigger.subs(k_sym, k_conc_sym))
        if ev.delay is not None:
            ev.delay = expr(ev.delay.subs(k_sym, k_conc_sym))
        if ev.priority is not None:
            ev.priority = expr(ev.priority.subs(k_sym, k_conc_sym))
        new_assignments = {}
        for var, assign_expr in ev.assignments.items():
            new_expr = expr(assign_expr.subs(k_sym, k_conc_sym))
            if var == k:
                if compartment in ev.assignments:
                    # Both species and compartment assigned: amount = formula*C_trigger,
                    # concentration = amount / C_new = formula * C_trigger / C_new_formula
                    comp_new = ev.assignments[compartment]
                    new_expr = expr(new_expr * comp_sym / comp_new)
                new_assignments[k_conc] = new_expr
            else:
                new_assignments[var] = new_expr
        ev.assignments = new_assignments


def _handle_conc_has_substance_units(
    k: str,
    compartment: str,
    init: sympy.Float,
    tmodel: data.Model,
) -> None:
    """Handle a species given in a concentration that is always intepreted as an amount.

    Sigh. Ok, so let's just multiply the concentration by the compartment and call it a
    day.

    """
    tmodel.variables[k] = data.Variable(value=init, unit=None)
    if (ia := tmodel.initial_assignments.get(k)) is not None:
        tmodel.initial_assignments[k] = _mul_expr(ia, compartment)
    else:
        tmodel.initial_assignments[k] = _mul_expr(init, compartment)

    # Fix initial assignment rule
    # Nothing to do here :)

    # Fix assignment rules
    # Nothing to do here :)

    # Fix rate rule
    # Nothing to do here :)

    # Fix reactions
    # Nothing to do here :)


def _handle_conc_boundary_has_substance_units(
    k: str,
    compartment: str,
    init: sympy.Float,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Handle a species in concentration units with both boundary=True and has_substance_units=True.

    Multiplies the concentration by the compartment to convert to amount.
    """
    tmodel.variables[k] = data.Variable(value=init, unit=None)
    if (ia := tmodel.initial_assignments.get(k)) is not None:
        tmodel.initial_assignments[k] = _mul_expr(ia, compartment)
    else:
        tmodel.initial_assignments[k] = _mul_expr(init, compartment)

    # Fix initial assignment rule
    # Nothing to do here :)

    # Fix assignment rules
    for dname in ctx.ass_rules_by_var[k]:
        tmodel.derived[dname] = _mul_expr(tmodel.derived[dname], compartment)

    # Fix rate rule
    # Nothing to do here :)

    # Fix reactions
    # Nothing to do here :)


def _transform_species(
    k: str,
    species: pdata.Species,
    pmodel: pdata.Model,
    tmodel: data.Model,
    ctx: Ctx,
) -> None:
    """Separate a species into parameter or variable and fix all reaction/rule references.

    Substitutes the correct concentration or amount expression wherever the species
    identifier appears in reactions, rules, and initial assignments.
    """
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
            return _handle_constant_variable(k=k, init=init, tmodel=tmodel, ctx=ctx)

        tmodel.variables[k] = data.Variable(value=init, unit=None)
        return None

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
                _handle_amount_boundary_has_substance_units(
                    k=k,
                    compartment=compartment,
                    init=init,
                    pmodel=pmodel,
                    tmodel=tmodel,
                    ctx=ctx,
                )

            else:
                LOGGER.debug("Species %s amount | True | False", k)
                _handle_amount_has_substance_units(
                    k=k,
                    init=init,
                    tmodel=tmodel,
                )

        else:  # noqa: PLR5501
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: amount | False | True", k)
                _handle_amount_boundary(
                    k=k,
                    compartment=compartment,
                    init=init,
                    pmodel=pmodel,
                    tmodel=tmodel,
                    ctx=ctx,
                )

            else:
                LOGGER.debug("Species %s: amount | False | False", k)
                _handle_amount(
                    k=k,
                    compartment=compartment,
                    init=init,
                    tmodel=tmodel,
                    ctx=ctx,
                )

    # We have a concentration here
    elif species.initial_concentration is not None:
        # We can always do this safely here, as we don't need any further transformation
        if variable_is_constant(k, pmodel):
            return _handle_constant_variable(k=k, init=init, tmodel=tmodel, ctx=ctx)

        # If it IS a concentration but has only substance units
        # is set, we have to multiply it by the compartment initially
        if species.has_only_substance_units:
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: | conc | True | True", k)
                _handle_conc_boundary_has_substance_units(
                    k=k,
                    compartment=compartment,
                    init=init,
                    tmodel=tmodel,
                    ctx=ctx,
                )

            else:
                LOGGER.debug("Species %s: | conc | True | False", k)
                _handle_conc_has_substance_units(
                    k=k,
                    compartment=compartment,
                    init=init,
                    tmodel=tmodel,
                )

        else:  # noqa: PLR5501
            if species.has_boundary_condition:
                LOGGER.debug("Species %s: | conc | False | True", k)
                _handle_conc_boundary(
                    k=k,
                    compartment=compartment,
                    init=init,
                    tmodel=tmodel,
                    ctx=ctx,
                )

            else:
                LOGGER.debug("Species %s: | conc | False | False", k)
                _handle_conc(
                    k=k,
                    compartment=compartment,
                    init=init,
                    pmodel=pmodel,
                    tmodel=tmodel,
                    ctx=ctx,
                )

    # Now BOTH of them are None, the whackest case of them all. If you think you can
    # figure out if it is a concentration or amount just by looking at species
    # and compartments, boy do I have a surprise for you :)
    # The documentation states you can use species.has_only_substance_units for this
    # which is false, as test cases 676 and 688 demonstrate. There S1 sets this as false
    # still it is being used as a concentration or an amount respectively
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
            # test 676 assumes amount
            # test 1513 assumes concentration??
            pmodel.variables[k].initial_amount = 0.0
            _transform_species(k, species, pmodel, tmodel, ctx)

    return None


def transform_species(pmodel: pdata.Model, tmodel: data.Model, ctx: Ctx) -> None:
    """Transform all species in the parsed model into the appropriate transformed model form."""
    LOGGER.debug("Species name | type | only subs. | boundary cond.")
    for k, var in pmodel.variables.items():
        _transform_species(k, var, pmodel, tmodel, ctx=ctx)


def _find_algebraic_floating_var(
    body_expr: sympy.Expr,
    pmodel: pdata.Model,
) -> str | None:
    """Return the variable name that an algebraic rule uniquely determines.

    The floating variable is a non-constant quantity that is not already
    determined by an assignment rule, rate rule, or reaction stoichiometry.
    """
    free = {s.name for s in body_expr.free_symbols if isinstance(s, sympy.Symbol)}

    determined: set[str] = set(pmodel.assignment_rules) | set(pmodel.rate_rules)
    for rxn in pmodel.reactions.values():
        for sp_name, stoich in rxn.stoichiometry.items():
            if isinstance(stoich, list) or stoich != 0:
                determined.add(sp_name)

    candidates: set[str] = set()
    for name in free:
        if (
            (name in pmodel.parameters and not pmodel.parameters[name].is_constant)
            or (name in pmodel.variables and not pmodel.variables[name].is_constant)
            or (
                name in pmodel.compartments
                and not pmodel.compartments[name].is_constant
            )
        ):
            candidates.add(name)

    candidates -= determined
    return next(iter(candidates)) if len(candidates) == 1 else None


def convert_algebraic_rules(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Resolve each algebraic rule by identifying and substituting its floating variable.

    Must be called *after* transform_species so concentration aliases (S_conc) are known.
    """
    for rule_name, rule in pmodel.algebraic_rules.items():
        body_expr = convert_mathml(rule.body, fns=tmodel.functions)

        floating_var = _find_algebraic_floating_var(body_expr, pmodel)
        if floating_var is None:
            LOGGER.warning(
                "Could not identify floating variable in algebraic rule %s; skipping",
                rule_name,
            )
            continue

        # Build substitutions: species that were transformed to amount form use
        # their _conc alias in rule expressions (SBML species = concentration there).
        conc_subs: dict[sympy.Symbol, sympy.Symbol] = {}
        for sym_name in (
            s.name for s in body_expr.free_symbols if isinstance(s, sympy.Symbol)
        ):
            if sym_name != floating_var and f"{sym_name}_conc" in tmodel.derived:
                conc_subs[sympy.Symbol(sym_name)] = sympy.Symbol(f"{sym_name}_conc")

        float_has_conc = f"{floating_var}_conc" in tmodel.derived
        if float_has_conc:
            float_solve_sym = sympy.Symbol(f"{floating_var}_conc")
            body_to_solve = body_expr.subs(conc_subs).subs(
                sympy.Symbol(floating_var), float_solve_sym
            )
        else:
            float_solve_sym = sympy.Symbol(floating_var)
            body_to_solve = expr(body_expr.subs(conc_subs))

        solutions = sympy.solve(body_to_solve, float_solve_sym)
        if not solutions:
            LOGGER.warning(
                "Could not solve algebraic rule %s for %s; skipping",
                rule_name,
                floating_var,
            )
            continue

        solution = cast(sympy.Expr, solutions[0])

        if float_has_conc:
            # Replace the placeholder _conc derived expression with the solved one.
            tmodel.derived[f"{floating_var}_conc"] = solution
            compartment = cast(str, pmodel.variables[floating_var].compartment)
            tmodel.derived[floating_var] = _mul_expr(solution, compartment)
        else:
            tmodel.derived[floating_var] = solution

        tmodel.variables.pop(floating_var, None)
        tmodel.parameters.pop(floating_var, None)


def apply_conversion_factors(pmodel: pdata.Model, tmodel: data.Model, ctx: Ctx) -> None:
    """Multiply reaction stoichiometries by per-species conversion factors."""
    model_cf = pmodel.conversion_factor
    for k, species in pmodel.variables.items():
        cf_name = species.conversion_factor or model_cf
        if cf_name is None:
            continue
        cf_sym = sympy.Symbol(cf_name)
        for rxn_name in ctx.rxns_by_var[k]:
            rxn = tmodel.reactions[rxn_name]
            for key in (k, f"{k}_amount"):
                if key in rxn.stoichiometry:
                    rxn.stoichiometry[key] = _mul_expr(rxn.stoichiometry[key], cf_sym)


def substitute_rate_of(tmodel: data.Model) -> None:
    """Replace __rateOf_x__ sentinel symbols with the actual dx/dt expressions."""
    rate_exprs: dict[str, sympy.Expr] = {}
    for rxn in tmodel.reactions.values():
        for var, stoich in rxn.stoichiometry.items():
            rate_exprs[var] = cast(
                sympy.Expr,
                rate_exprs.get(var, sympy.Float(0.0)) + _mul_expr(stoich, rxn.expr),
            )

    # Constant parameters and variables with no reactions → rate = 0
    for name in list(tmodel.parameters) + list(tmodel.variables):
        if name not in rate_exprs:
            rate_exprs[name] = sympy.Float(0.0)

    subs_dict = {
        sympy.Symbol(f"__rateOf_{var}__"): rate_expr
        for var, rate_expr in rate_exprs.items()
    }

    # Any remaining __rateOf_*__ sentinels (e.g. local params) → rate = 0
    def _collect_all_exprs() -> list[sympy.Expr]:
        result: list[sympy.Expr] = []
        for rxn in tmodel.reactions.values():
            result.append(rxn.expr)
        for e in tmodel.derived.values():
            result.append(e)
        for e in tmodel.initial_assignments.values():
            result.append(e)
        for ev in tmodel.events.values():
            if ev.trigger is not None:
                result.append(ev.trigger)
            result.extend(ev.assignments.values())
            if ev.delay is not None:
                result.append(ev.delay)
            if ev.priority is not None:
                result.append(ev.priority)
        return result

    for e in _collect_all_exprs():
        for sym in e.free_symbols:
            if (
                isinstance(sym, sympy.Symbol)
                and sym.name.startswith("__rateOf_")
                and sym not in subs_dict
            ):
                subs_dict[sym] = sympy.Float(0.0)

    for rxn in tmodel.reactions.values():
        rxn.expr = expr(rxn.expr.subs(subs_dict))

    for name in list(tmodel.derived):
        tmodel.derived[name] = expr(tmodel.derived[name].subs(subs_dict))

    for name in list(tmodel.initial_assignments):
        tmodel.initial_assignments[name] = expr(
            tmodel.initial_assignments[name].subs(subs_dict)
        )

    for ev in tmodel.events.values():
        if ev.trigger is not None:
            ev.trigger = expr(ev.trigger.subs(subs_dict))
        ev.assignments = {
            var: expr(assign_expr.subs(subs_dict))
            for var, assign_expr in ev.assignments.items()
        }
        if ev.delay is not None:
            ev.delay = expr(ev.delay.subs(subs_dict))
        if ev.priority is not None:
            ev.priority = expr(ev.priority.subs(subs_dict))


def substitute_delays(tmodel: data.Model) -> None:
    """Replace SBMLDelay(x, d) sentinels with time-shifted expressions where possible.

    - Assignment rules: substitute time → time - d directly (exact).
    - Variables with initial assignment but no dynamics: piecewise(history, t<d, x).
    - Fallback: return current value (ignores delay, approximate).
    """
    time_sym = sympy.Symbol("time")

    has_dynamics: set[str] = {
        var for rxn in tmodel.reactions.values() for var in rxn.stoichiometry
    }

    def resolve(target_expr: sympy.Expr, delay_amt: sympy.Expr) -> sympy.Expr:
        if not isinstance(target_expr, sympy.Symbol):
            return target_expr
        name = target_expr.name

        if name in tmodel.derived:
            return expr(tmodel.derived[name].subs(time_sym, time_sym - delay_amt))

        if name in tmodel.initial_assignments and name not in has_dynamics:
            history = expr(
                tmodel.initial_assignments[name].subs(time_sym, time_sym - delay_amt)
            )
            return expr(
                sympy.Piecewise((history, time_sym < delay_amt), (target_expr, True))
            )

        return target_expr

    def _sub(e: sympy.Expr) -> sympy.Expr:
        if e.has(SBMLDelay):
            return expr(e.replace(SBMLDelay, resolve))
        return e

    for name in list(tmodel.derived):
        tmodel.derived[name] = _sub(tmodel.derived[name])
    for name in list(tmodel.initial_assignments):
        tmodel.initial_assignments[name] = _sub(tmodel.initial_assignments[name])
    for rxn in tmodel.reactions.values():
        rxn.expr = _sub(rxn.expr)
    for ev in tmodel.events.values():
        if ev.trigger is not None:
            ev.trigger = _sub(ev.trigger)
        ev.assignments = {var: _sub(e) for var, e in ev.assignments.items()}
        if ev.delay is not None:
            ev.delay = _sub(ev.delay)
        if ev.priority is not None:
            ev.priority = _sub(ev.priority)


def transform(doc: pdata.Document) -> data.Model:
    """Transform a parsed SBML document into a simplified sympy-based model."""
    pmodel = doc.model

    ctx = Ctx(rxns_by_var=defaultdict(set), ass_rules_by_var=defaultdict(set))
    for name, rxn in pmodel.reactions.items():
        for arg in rxn.args:
            ctx.rxns_by_var[arg.name].add(name)
        for arg in rxn.stoichiometry:
            ctx.rxns_by_var[arg].add(name)
    for name, rule in pmodel.assignment_rules.items():
        ctx.ass_rules_by_var[name].add(name)
        for arg in rule.args:
            ctx.ass_rules_by_var[arg.name].add(name)

    tmodel = data.Model(name=pmodel.name)  # type: ignore
    convert_units(pmodel=pmodel, tmodel=tmodel)
    convert_parameters(pmodel=pmodel, tmodel=tmodel)
    convert_compartments(pmodel=pmodel, tmodel=tmodel)
    convert_constraints(pmodel=pmodel, tmodel=tmodel)
    convert_functions(pmodel=pmodel, tmodel=tmodel)
    convert_events(pmodel=pmodel, tmodel=tmodel)
    convert_rules_and_initial_assignments(pmodel=pmodel, tmodel=tmodel)
    convert_reactions(pmodel=pmodel, tmodel=tmodel)

    # Do the heavy lifting here
    transform_species(pmodel=pmodel, tmodel=tmodel, ctx=ctx)
    apply_conversion_factors(pmodel=pmodel, tmodel=tmodel, ctx=ctx)
    remove_duplicate_entries(tmodel=tmodel)
    convert_algebraic_rules(pmodel=pmodel, tmodel=tmodel)
    substitute_rate_of(tmodel=tmodel)
    substitute_delays(tmodel=tmodel)

    # Event-assigned parameters must be variables so the ODE state tracks changes
    event_targets = {var for ev in tmodel.events.values() for var in ev.assignments}
    for name in list(event_targets):
        if name in tmodel.parameters:
            par = tmodel.parameters.pop(name)
            tmodel.variables[name] = data.Variable(value=par.value, unit=par.unit)

    return tmodel
