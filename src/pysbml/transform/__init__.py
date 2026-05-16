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
import warnings
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
    rxn_had_compartment: dict[str, bool]
    # Deferred stoichiometry corrections: (rxn_name, species_k, compartment)
    # Applied after all species are transformed so rxn_had_compartment is final.
    pending_stoich_corrections: list[tuple[str, str, str]]


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
        if name in pmodel.rate_rules:
            return False
        # Event assignments can change the species; must not treat as constant
        return not any(
            any(a.variable == name for a in ev.assignments)
            for ev in pmodel.events.values()
        )
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
        if par.is_constant and k not in pmodel.rate_rules:
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
            existing = tmodel.parameters.pop(name, None)
            init = existing.value if existing is not None else sympy.Float(0.0)
            tmodel.variables[name] = data.Variable(value=init, unit=None)

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
        # rateOf(local_param) sentinels also need renaming so substitute_rate_of
        # resolves them to 0 (local params are always constant).
        pars_to_replace.update(
            {f"__rateOf_{pn}__": f"__rateOf_{name}_{pn}__" for pn in rxn.local_pars}
        )
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
    if k in tmodel.derived:
        tmodel.derived[k] = _mul_expr(tmodel.derived[k], compartment)
        tmodel.derived[f"{k}_conc"] = _div_expr(k, compartment)

    # Fix rate rule: d(S_amount)/dt = d(S_conc)/dt * C + S_conc * d(C)/dt
    if (rr := tmodel.reactions.get(f"d{k}")) is not None:
        comp_rate_rxn = tmodel.reactions.get(f"d{compartment}")
        if comp_rate_rxn is not None:
            conc_sym = sympy.Symbol(k_conc)
            rr.expr = expr(
                _mul_expr(rr.expr, compartment)
                + _mul_expr(conc_sym, comp_rate_rxn.expr)
            )
            rr.stoichiometry = {k: sympy.Float(1.0)}
        else:
            rr.stoichiometry = {k: sympy.Symbol(compartment)}

    # Fix reactions (defer stoichiometry correction until all species are processed)
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]
        rxn.expr = expr(rxn.expr.subs({compartment: 1, k: k_conc}))
        ctx.pending_stoich_corrections.append((rxn_name, k, compartment))

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

    # Fix assignment rule on the species itself: the rule assigns concentration,
    # while reported species values are amounts for these fixtures.
    if k in tmodel.derived:
        tmodel.derived[k] = _mul_expr(tmodel.derived[k], compartment)
        tmodel.derived[f"{k}_conc"] = _div_expr(k, compartment)

    # Fix derived (assignment) rules: k appears as concentration in math
    k_sym = sympy.Symbol(k)
    k_conc_sym = sympy.Symbol(k_conc)
    for dname in ctx.ass_rules_by_var[k]:
        if dname != k:
            tmodel.derived[dname] = expr(tmodel.derived[dname].subs(k_sym, k_conc_sym))

    # Fix rate rule: d(S_amount)/dt = d(S_conc)/dt * C + S_conc * d(C)/dt
    if (rr := tmodel.reactions.get(f"d{k}")) is not None:
        rr.expr = expr(rr.expr.subs(k_sym, k_conc_sym))
        comp_rate_rxn = tmodel.reactions.get(f"d{compartment}")
        if comp_rate_rxn is not None:
            rr.expr = expr(
                _mul_expr(rr.expr, compartment)
                + _mul_expr(k_conc_sym, comp_rate_rxn.expr)
            )
            rr.stoichiometry = {k: sympy.Float(1.0)}
        else:
            rr.stoichiometry = {k: sympy.Symbol(compartment)}

    # Fix reactions: substitute boundary species with its concentration symbol.
    # Any compartment factor in the kinetic law is handled by _handle_amount for
    # non-boundary species in the same reaction.
    k_sym_local = sympy.Symbol(k)
    k_conc_sym_local = sympy.Symbol(k_conc)
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]
        rxn.expr = expr(rxn.expr.subs(k_sym_local, k_conc_sym_local))
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
    compartment: str,
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
    tmodel.derived[f"{k}_conc"] = _div_expr(k, compartment)
    tmodel.substance_units_vars = tmodel.substance_units_vars | frozenset([k])

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

    # Fix reactions: k_conc = k/compartment introduces compartment dependency
    for rxn_name in ctx.rxns_by_var[k]:
        rxn = tmodel.reactions[rxn_name]
        rxn.expr = expr(rxn.expr.subs(k, k_conc))
        ctx.rxn_had_compartment[rxn_name] = True


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
        tmodel.derived[f"{k}_amount"] = _mul_expr(
            sympy.Symbol(k), sympy.Symbol(compartment)
        )

        # Fix initial assignment rule
        if (comp := tmodel.initial_assignments.get(compartment)) is not None:
            tmodel.initial_assignments[k] = _mul_expr(init, comp)

        # Fix assignment rule
        # Nothing to do here :)

        # Fix rate rule
        # Nothing to do here :)

        # Fix reactions: d(conc)/dt = stoich * v / V, so normalize stoich by compartment
        for rxn_name in ctx.rxns_by_var[k]:
            rxn = tmodel.reactions[rxn_name]
            if k in rxn.stoichiometry:
                rxn.stoichiometry[k] = _div_expr(rxn.stoichiometry[k], compartment)
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
        # If compartment is assigned but species is not, conserve amount:
        # S_conc_new = S_conc_old * C_old / C_new
        if compartment in new_assignments and k_conc not in new_assignments:
            comp_new = new_assignments[compartment]
            new_assignments[k_conc] = expr(k_conc_sym * comp_sym / comp_new)
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

    # Fix assignment rules (only those without rateOf sentinels — rateOf already in
    # amount/time units and must not be scaled by compartment)
    for dname in ctx.ass_rules_by_var[k]:
        rule_expr = tmodel.derived[dname]
        has_rate_of = any(
            isinstance(s, sympy.Symbol) and s.name.startswith("__rateOf_")
            for s in rule_expr.free_symbols
        )
        if not has_rate_of:
            tmodel.derived[dname] = _mul_expr(rule_expr, compartment)

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
                    compartment=compartment,
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
        if variable_is_constant(k, pmodel):
            comp_obj = pmodel.compartments[compartment]
            if not species.has_only_substance_units and not comp_obj.is_constant:
                # D13 fix: constant=true + HOSU=false + initialConcentration + dynamic
                # compartment. Spec §4.6.4: the AMOUNT is held constant, not the
                # concentration. Derive k = k_amount / C(t) so concentration tracks the
                # compartment while the amount stays fixed.
                k_amount = f"{k}_amount"
                tmodel.parameters[k_amount] = data.Parameter(
                    value=sympy.Float(float(init) * comp_obj.size), unit=None
                )
                tmodel.derived[k] = _div_expr(k_amount, compartment)
                for name in ctx.rxns_by_var[k]:
                    tmodel.reactions[name].stoichiometry.pop(k, None)
            else:
                _handle_constant_variable(k=k, init=init, tmodel=tmodel, ctx=ctx)
            return None

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

    # Apply deferred stoichiometry corrections: multiply by compartment only when the
    # (final, fully-transformed) kinetic law has a concentration/time dependency.
    for rxn_name, species_k, compartment in ctx.pending_stoich_corrections:
        if ctx.rxn_had_compartment.get(rxn_name, False):
            rxn = tmodel.reactions.get(rxn_name)
            if rxn is not None and (s := rxn.stoichiometry.get(species_k)) is not None:
                rxn.stoichiometry[species_k] = _mul_expr(s, compartment)


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
    if len(candidates) == 1:
        return next(iter(candidates))

    # L1v2 fallback: libsbml marks all L1 parameters/compartments constant=True by
    # default. If no non-constant candidate was found, try parameters and compartments
    # that are not otherwise determined — in a well-formed L1 model exactly one is
    # floating.
    if not candidates:
        l1_fallback = {
            name
            for name in free
            if (name in pmodel.parameters or name in pmodel.compartments)
            and name not in determined
        }
        if len(l1_fallback) == 1:
            return next(iter(l1_fallback))

    return None


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


def convert_fast_reactions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    """Apply QSS reduction: species exclusively in fast reactions become derived."""
    fast_rxn_names = {name for name, rxn in pmodel.reactions.items() if rxn.fast}
    if not fast_rxn_names:
        return

    init_param_subs: dict[sympy.Symbol, sympy.Float] = {
        sympy.Symbol(n): sympy.Float(p.value) for n, p in pmodel.parameters.items()
    }
    for n, c in pmodel.compartments.items():
        init_param_subs[sympy.Symbol(n)] = sympy.Float(c.size)

    qss_species: set[str] = set()
    # Deferred: species whose fast reaction is inactive at t=0 (parameter-gated).
    # Value: (net_flux expr, QSS solution expr) for event assignment generation.
    deferred_qss: dict[str, tuple[sympy.Expr, sympy.Expr]] = {}

    for species in list(tmodel.variables):
        participating = {
            name
            for name, rxn in pmodel.reactions.items()
            if species in rxn.stoichiometry
        }
        if not participating or not participating.issubset(fast_rxn_names):
            continue

        net_flux: sympy.Expr = sympy.Float(0.0)
        for rxn in tmodel.reactions.values():
            s = rxn.stoichiometry.get(species)
            if s is not None:
                net_flux = sympy.Add(net_flux, _mul_expr(s, rxn.expr))

        species_conc = f"{species}_conc"
        has_conc = species_conc in tmodel.derived
        solve_sym = sympy.Symbol(species_conc if has_conc else species)
        if has_conc:
            net_flux = net_flux.subs(sympy.Symbol(species), solve_sym)

        solutions = sympy.solve(net_flux, solve_sym)
        if not solutions:
            msg = f"QSS: cannot solve algebraically for {species}"
            raise NotImplementedError(msg)
        if len(solutions) > 1:
            warnings.warn(
                f"Multiple QSS solutions for {species}; using first", stacklevel=2
            )

        solution = cast(sympy.Expr, solutions[0])

        # Piecewise kinetics may produce a solution with NaN branches (sympy
        # solves one branch but not the other). Conservation-law-based QSS
        # is needed for these, which we don't support yet.
        if solution.has(sympy.S.NaN):
            msg = f"QSS: piecewise solution with nan branch for {species}"
            raise NotImplementedError(msg)

        # If the fast flux is identically zero at t=0 (e.g. a rate parameter starts
        # at 0 and is set by an event), QSS is not valid from t=0. Keep the species
        # as a state variable and handle equilibration via event assignments instead.
        try:
            is_deferred = bool(net_flux.subs(init_param_subs).is_zero)
        except (TypeError, ValueError):
            is_deferred = False
        if is_deferred:
            deferred_qss[species] = (net_flux, solution)
            continue

        if has_conc:
            compartment = cast(str, pmodel.variables[species].compartment)
            tmodel.derived[species_conc] = solution
            tmodel.derived[species] = _mul_expr(solution, compartment)
        else:
            tmodel.derived[species] = solution
        del tmodel.variables[species]
        qss_species.add(species)

    if qss_species:
        _apply_qss_corrections(pmodel, tmodel, fast_rxn_names, qss_species)

    if deferred_qss:
        _apply_deferred_qss_events(pmodel, tmodel, fast_rxn_names, deferred_qss)


def _apply_qss_corrections(
    pmodel: pdata.Model,
    tmodel: data.Model,
    fast_rxn_names: set[str],
    qss_species: set[str],
) -> None:
    """Correct ODE stoichiometries and initial conditions after QSS reduction.

    For non-QSS species S_j in the fast subsystem, the effective stoichiometry
    in each reaction R is T_R / (dT/dS_j), where T_R = c^T * stoich_R and
    dT/dS_j = c_j + sum_{k in QSS} c_k * d(k_amount)/d(S_j), with c the
    conservation vector from null(N_fast^T).
    """
    fast_species_all: set[str] = set()
    for rn in fast_rxn_names:
        fast_species_all.update(pmodel.reactions[rn].stoichiometry.keys())

    non_qss_in_fast = fast_species_all & set(tmodel.variables)
    fast_sp_list = sorted(fast_species_all)
    fast_rxn_list = sorted(fast_rxn_names)

    # Build stoichiometry matrix for fast reactions (amounts, from pmodel)
    N_rows = []
    for sp in fast_sp_list:
        row = []
        for rn in fast_rxn_list:
            st = pmodel.reactions[rn].stoichiometry.get(sp, 0.0)
            if isinstance(st, list):
                st = sum(s[0] for s in st)
            row.append(float(st))
        N_rows.append(row)
    N_mat = sympy.Matrix(N_rows)
    conservation_vecs = N_mat.T.nullspace()

    if conservation_vecs and non_qss_in_fast:
        c_vec = conservation_vecs[0]
        c_map = {fast_sp_list[i]: float(c_vec[i]) for i in range(len(fast_sp_list))}

        # Substitution: _conc symbols → amount/comp, for differentiating QSS exprs
        conc_to_amount: dict[sympy.Symbol, sympy.Expr] = {}
        for sp in non_qss_in_fast:
            v = pmodel.variables.get(sp)
            if v is not None and v.compartment:
                comp_sym = sympy.Symbol(v.compartment)
                conc_to_amount[sympy.Symbol(f"{sp}_conc")] = sympy.Symbol(sp) / comp_sym

        # QSS amount expressions in amount-space (for differentiation)
        qss_amount_exprs: dict[str, sympy.Expr] = {}
        for sp_k in qss_species:
            qss_amount_exprs[sp_k] = cast(
                sympy.Expr, tmodel.derived[sp_k].subs(conc_to_amount)
            )

        # dT/dS_j for each non-QSS species
        dt_ds: dict[str, sympy.Expr] = {}
        for sp_j in non_qss_in_fast:
            sym_j = sympy.Symbol(sp_j)
            val: sympy.Expr = sympy.Float(c_map[sp_j])
            for sp_k in qss_species:
                c_k = c_map.get(sp_k, 0.0)
                if c_k != 0.0:
                    val = cast(
                        sympy.Expr,
                        val + c_k * sympy.diff(qss_amount_exprs[sp_k], sym_j),
                    )
            dt_ds[sp_j] = val

        # Compute corrected stoichiometries (before removing QSS species)
        corrections: dict[str, dict[str, sympy.Expr]] = {}
        for rn, rxn in tmodel.reactions.items():
            corr: dict[str, sympy.Expr] = {}
            for sp_j in non_qss_in_fast:
                t_r: sympy.Expr = sympy.Float(0.0)
                for sp_k in fast_sp_list:
                    c_k = c_map.get(sp_k, 0.0)
                    if c_k == 0.0:
                        continue
                    st = rxn.stoichiometry.get(sp_k)
                    if st is None or isinstance(st, list):
                        continue
                    st_e = st if isinstance(st, sympy.Expr) else sympy.Float(float(st))
                    t_r = cast(sympy.Expr, t_r + c_k * st_e)
                denom = dt_ds[sp_j]
                eff = (
                    sympy.cancel(t_r / denom) if not denom.is_zero else sympy.Float(0.0)
                )
                corr[sp_j] = eff
            corrections[rn] = corr

        # Apply corrected stoichiometries
        for rn, rxn in tmodel.reactions.items():
            for sp_j, eff in corrections[rn].items():
                if eff.is_zero:
                    rxn.stoichiometry.pop(sp_j, None)
                else:
                    rxn.stoichiometry[sp_j] = eff

    # Remove QSS species from all reaction stoichiometries
    for rxn in tmodel.reactions.values():
        for sp in qss_species:
            rxn.stoichiometry.pop(sp, None)

    if not non_qss_in_fast or not conservation_vecs:
        return

    # Fix initial conditions: project non-QSS species onto QSS manifold at t=0
    def init_amount(sp: str) -> float:
        v = pmodel.variables.get(sp)
        if v is None:
            return 0.0
        if v.initial_amount:
            return float(v.initial_amount)
        comp = v.compartment
        sz = (
            pmodel.compartments[comp].size
            if comp and comp in pmodel.compartments
            else 1.0
        )
        return float(v.initial_concentration or 0.0) * sz

    sym_map = {sp: sympy.Symbol(sp) for sp in fast_sp_list}

    conservation_eqs = []
    for c_vec in conservation_vecs:
        total = sympy.Float(
            sum(
                float(c_vec[i]) * init_amount(fast_sp_list[i])
                for i in range(len(fast_sp_list))
            )
        )
        lhs = sum(c_vec[i] * sym_map[fast_sp_list[i]] for i in range(len(fast_sp_list)))
        conservation_eqs.append(sympy.Eq(lhs, total))

    qss_eqs = []
    for sp in qss_species:
        if sp not in fast_species_all:
            continue
        qss_expr_raw = tmodel.derived[sp]
        subs: dict[sympy.Symbol, sympy.Expr] = {}
        for other_sp in fast_species_all:
            v = pmodel.variables.get(other_sp)
            if v is None:
                continue
            comp = v.compartment
            sz = sympy.Float(
                pmodel.compartments[comp].size
                if comp and comp in pmodel.compartments
                else 1.0
            )
            subs[sympy.Symbol(f"{other_sp}_conc")] = sym_map[other_sp] / sz
            if comp:
                subs[sympy.Symbol(comp)] = sz
        qss_eqs.append(sympy.Eq(sym_map[sp], expr(qss_expr_raw.subs(subs))))

    all_sym_list = [sym_map[sp] for sp in fast_sp_list]
    solutions = sympy.solve(conservation_eqs + qss_eqs, all_sym_list)
    if not solutions:
        return

    for sp in sorted(non_qss_in_fast):
        sol_expr = solutions.get(sym_map[sp])
        if sol_expr is not None and sp in tmodel.variables:
            tmodel.initial_assignments[sp] = expr(sol_expr)


def _apply_deferred_qss_events(
    pmodel: pdata.Model,
    tmodel: data.Model,
    fast_rxn_names: set[str],
    deferred_qss: dict[str, tuple[sympy.Expr, sympy.Expr]],
) -> None:
    """Add event assignments for fast species whose QSS is inactive at t=0.

    When a fast reaction's rate is zero at t=0 (e.g. parameter starts at 0),
    the species cannot be derived from t=0. Instead, when an event activates
    the rate, we inject event assignments that:
      - set the QSS species to its equilibrium value (e.g. 0)
      - update conserved non-QSS species by the displaced mass
    """
    fast_species_all: set[str] = set()
    for rn in fast_rxn_names:
        fast_species_all.update(pmodel.reactions[rn].stoichiometry.keys())

    fast_sp_list = sorted(fast_species_all)
    fast_rxn_list = sorted(fast_rxn_names)

    N_rows = []
    for sp in fast_sp_list:
        row = []
        for rn in fast_rxn_list:
            st = pmodel.reactions[rn].stoichiometry.get(sp, 0.0)
            if isinstance(st, list):
                st = sum(s[0] for s in st)
            row.append(float(st))
        N_rows.append(row)
    N_mat = sympy.Matrix(N_rows)
    conservation_vecs = N_mat.T.nullspace()

    c_map: dict[str, float] = {}
    if conservation_vecs:
        c_vec = conservation_vecs[0]
        c_map = {fast_sp_list[i]: float(c_vec[i]) for i in range(len(fast_sp_list))}

    non_qss_in_fast = fast_species_all & set(tmodel.variables) - set(deferred_qss)

    for species, (net_flux, solution) in deferred_qss.items():
        flux_params = net_flux.free_symbols & {
            sympy.Symbol(n) for n in pmodel.parameters
        }
        for event in tmodel.events.values():
            if not any(str(p) in event.assignments for p in flux_params):
                continue
            # Only inject QSS assignment when the event ACTIVATES the fast reaction.
            # If the new parameter value makes net_flux structurally zero, this
            # event deactivates the reaction — don't force the QSS value.
            new_param_subs: dict[sympy.Symbol, sympy.Expr] = {
                p: cast(sympy.Expr, event.assignments[str(p)])
                for p in flux_params
                if str(p) in event.assignments
            }
            try:
                activates = not bool(net_flux.subs(new_param_subs).is_zero)
            except (TypeError, ValueError):
                activates = True
            if not activates:
                continue
            event.assignments[species] = cast(sympy.Expr, solution)
            c_k = c_map.get(species, 0.0)
            if c_k == 0.0:
                continue
            for sp_j in sorted(non_qss_in_fast):
                c_j = c_map.get(sp_j, 0.0)
                if c_j == 0.0:
                    continue
                # sp_j_new = sp_j_old + (c_k/c_j) * (species_old - solution)
                # species_old is Symbol(species), evaluated pre-assignment
                mass_transfer = cast(
                    sympy.Expr,
                    sympy.Rational(round(c_k), round(c_j))
                    * (sympy.Symbol(species) - solution),
                )
                existing = cast(
                    sympy.Expr,
                    event.assignments.get(sp_j, sympy.Symbol(sp_j)),
                )
                event.assignments[sp_j] = cast(
                    sympy.Expr, sympy.Add(existing, mass_transfer)
                )


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

    # Build rateOf substitution dict.
    # rateOf(S) should give the rate-of-change of S as it appears in math expressions:
    # - amount-tracked (S in variables, S_conc in derived): rateOf = amount_rate / C
    # - conc-tracked boundary (S in derived, S_conc in variables): rateOf = rate_of_S_conc
    # - direct tracking: rateOf = rate_expr
    # hasOnlySubstanceUnits=True vars are excluded from the concentration-rate formula
    # because their identifier already means amount in all expressions.
    subs_dict = {}
    for var, rate_expr in rate_exprs.items():
        conc_name = f"{var}_conc"
        if conc_name in tmodel.derived and var not in tmodel.substance_units_vars:
            conc_expr = tmodel.derived[conc_name]
            var_sym = sympy.Symbol(var)
            compartment_sym = sympy.cancel(var_sym / conc_expr)
            compartment_str = str(compartment_sym)
            dC_rate = rate_exprs.get(compartment_str, sympy.Float(0.0))
            conc_sym = sympy.Symbol(conc_name)
            # d(S_conc)/dt = rate_amount/C - S_conc/C * dC_rate
            subs_dict[sympy.Symbol(f"__rateOf_{var}__")] = expr(
                rate_expr / compartment_sym - conc_sym / compartment_sym * dC_rate
            )
        else:
            subs_dict[sympy.Symbol(f"__rateOf_{var}__")] = rate_expr

    # Conc-boundary species: S is in derived (S = S_conc * C), rateOf(S) = rate_of_S_conc
    for var in tmodel.derived:
        conc_name = f"{var}_conc"
        sentinel = sympy.Symbol(f"__rateOf_{var}__")
        if conc_name in rate_exprs and sentinel not in subs_dict:
            subs_dict[sentinel] = rate_exprs[conc_name]

    # Any remaining __rateOf_*__ sentinels (e.g. local params) → rate = 0
    def _collect_all_exprs() -> list[sympy.Expr]:
        result: list[sympy.Expr] = []
        for rxn in tmodel.reactions.values():
            result.append(rxn.expr)  # noqa: PERF401
        for e in tmodel.derived.values():
            result.append(e)  # noqa: PERF402  # noqa: PERF402
        for e in tmodel.initial_assignments.values():
            result.append(e)  # noqa: PERF402
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
            msg = "delay() on complex expressions not supported"
            raise NotImplementedError(msg)
        name = target_expr.name

        if name in tmodel.derived:
            orig = tmodel.derived[name]
            substituted = expr(orig.subs(time_sym, time_sym - delay_amt))
            if substituted == orig:
                dyn_in_expr = {
                    s.name for s in orig.free_symbols if isinstance(s, sympy.Symbol)
                } & has_dynamics
                if dyn_in_expr:
                    msg = f"delay({name}) on dynamic variable expression not supported (DDE)"
                    raise NotImplementedError(msg)
            return substituted

        if name in tmodel.initial_assignments and name not in has_dynamics:
            history = expr(
                tmodel.initial_assignments[name].subs(time_sym, time_sym - delay_amt)
            )
            return expr(
                sympy.Piecewise((history, time_sym < delay_amt), (target_expr, True))
            )

        if name in tmodel.parameters:
            return target_expr

        msg = f"delay({name}) for dynamic variable not supported (DDE)"
        raise NotImplementedError(msg)

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

    ctx = Ctx(
        rxns_by_var=defaultdict(set),
        ass_rules_by_var=defaultdict(set),
        rxn_had_compartment={},
        pending_stoich_corrections=[],
    )
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

    # Pre-compute: for each reaction, whether any compartment symbol was in the
    # original kinetic law. Used by _handle_amount to decide if stoichiometry
    # needs compensating C factor (L2-style) or not (L3-style).
    compartment_syms = {sympy.Symbol(c) for c in pmodel.compartments}
    for rxn_name, rxn in tmodel.reactions.items():
        ctx.rxn_had_compartment[rxn_name] = bool(
            rxn.expr.free_symbols & compartment_syms
        )

    # Do the heavy lifting here
    transform_species(pmodel=pmodel, tmodel=tmodel, ctx=ctx)
    apply_conversion_factors(pmodel=pmodel, tmodel=tmodel, ctx=ctx)
    remove_duplicate_entries(tmodel=tmodel)
    convert_algebraic_rules(pmodel=pmodel, tmodel=tmodel)
    convert_fast_reactions(pmodel=pmodel, tmodel=tmodel)
    substitute_rate_of(tmodel=tmodel)
    substitute_delays(tmodel=tmodel)

    # Event-assigned parameters must be variables so the ODE state tracks changes
    event_targets = {var for ev in tmodel.events.values() for var in ev.assignments}
    for name in list(event_targets):
        if name in tmodel.parameters:
            par = tmodel.parameters.pop(name)
            tmodel.variables[name] = data.Variable(value=par.value, unit=par.unit)

    return tmodel
