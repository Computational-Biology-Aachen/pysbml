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
from functools import reduce
from operator import mul
from typing import cast

import sympy

from pysbml.parse import data as pdata
from pysbml.transform.units import CONVERSION, PREFIXES

from . import data
from .mathml2sympy import convert_mathml

LOGGER = logging.getLogger(__name__)


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


def convert_species(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, var in pmodel.variables.items():
        if var.conversion_factor is not None:
            raise NotImplementedError

        if (init := var.initial_amount) is None and (
            init := var.initial_concentration
        ) is None:
            init = 0.0

        # Change species that are constant or boundary conditions
        # to parameters
        if variable_is_constant(k, pmodel):
            tmodel.parameters[k] = data.Parameter(value=sympy.Float(init), unit=None)
            continue

        # Check if it is an amount rather than a concentration
        if var.initial_concentration is None:
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(init), unit=None
            )
            continue

        # Another whack one
        if (
            var.initial_concentration is not None
            and compartment_is_valid(pmodel, var)
            and not var.has_only_substance_units
            and var.has_boundary_condition
        ):
            compartment = cast(str, var.compartment)
            tmodel.derived[f"{k}_amount"] = data.Derived(
                fn=_div_expr(k, compartment),
                args=[k, cast(str, compartment)],
            )
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(init), unit=None
            )
            continue

        # This one is whack. If it IS a concentration but has only substance units
        # is set, we have to multiply it by the compartment initially
        if (
            var.initial_concentration is not None
            and compartment_is_valid(pmodel, var)
            and var.has_only_substance_units
        ):
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(init), unit=None
            )
            tmodel.initial_assignments[k] = data.Derived(
                fn=init * sympy.Symbol(var.compartment),  # type: ignore
                args=[cast(str, var.compartment)],
            )
            continue

        # Ignore complexity if compartment isn't valid
        if not compartment_is_valid(pmodel, var):
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(init), unit=None
            )
            continue

        # Concentration means we need to keep track of the compartment size, because it
        # will change dynamically
        # Checked this isn't None via compartment_is_valid
        compartment = cast(str, var.compartment)
        tmodel.derived[k] = data.Derived(
            fn=_div_expr(f"{k}_amount", compartment),
            args=[f"{k}_amount", cast(str, compartment)],
        )
        tmodel.variables[f"{k}_amount"] = data.Variable(
            initial_value=sympy.Float(init), unit=None
        )

        if k not in pmodel.initial_assignments:
            tmodel.initial_assignments[f"{k}_amount"] = data.Derived(
                fn=init * sympy.Symbol(compartment),  # type: ignore
                args=[cast(str, compartment)],
            )
        else:
            # Remove and steal the expression
            ass = pmodel.initial_assignments.pop(k)
            fn = convert_mathml(ass.body, fns=tmodel.functions) * sympy.Symbol(
                compartment
            )  # type: ignore
            tmodel.initial_assignments[f"{k}_amount"] = data.Derived(
                fn=fn,
                args=[i.name for i in fn.args if isinstance(i, sympy.Symbol)],
            )


def _convert_rate_rule_arg_to_conc(
    name: str, pmodel: pdata.Model
) -> dict[str, sympy.Float | str]:
    # the parsed species part is important to not
    # introduce conversion on things that aren't species
    if (species := pmodel.variables.get(name)) is None:
        return {name: sympy.Float(1.0)}

    if (
        not species.is_concentration()
        and compartment_is_valid(pmodel, species)
        and not species.has_only_substance_units
    ):
        return {name: cast(str, species.compartment)}

    # Yes, there is a case where it has a concentration, but `has_only_substrate_units`
    # is `True`, which has to behave differently
    if (
        species.is_concentration()
        and compartment_is_valid(pmodel, species)
        and not species.has_only_substance_units
    ):
        if species.has_boundary_condition:
            return {name: cast(str, species.compartment)}
        return {f"{name}_amount": cast(str, species.compartment)}

    if (
        species.is_concentration()
        and compartment_is_valid(pmodel, species)
        and species.has_only_substance_units
    ):
        return {name: sympy.Float(1.0)}

    if species.has_only_substance_units and not species.has_boundary_condition:
        return {name: cast(str, species.compartment)}

    # Safe?
    return {name: sympy.Float(1.0)}


def _convert_init_target_to_conc(
    name: str, fn: sympy.Expr, args: list[str], pmodel: pdata.Model
) -> tuple[sympy.Expr, list[str]]:
    # the parsed species part is important to not
    # introduce conversion on things that aren't species
    if (species := pmodel.variables.get(name)) is None:
        return fn, args

    if (
        not species.is_concentration()
        and compartment_is_valid(pmodel, species)
        and species.initial_amount is not None
    ):
        fn = fn * sympy.Symbol(species.compartment)  # type: ignore
        return fn, free_symbols(fn)

    # There are also cases where BOTH are concentration and amount are None
    # In 676 one falls back to an amount, but in 688 one falls back to a concentration
    # Here is the kicker: this is not dependent on S1 AT ALL, but can only be infered
    # from S2. The two of them are linked by a reaction so apparenlty "have" to have
    # the same unit. Kill me.
    if species.initial_amount is None and species.initial_concentration is None:
        LOGGER.warning(
            "Neither initial amount nor initial concentration set for %s", name
        )
        is_amount = False

        for reaction in pmodel.reactions.values():
            if name in reaction.stoichiometry:
                for _name in reaction.stoichiometry:
                    if pmodel.variables[_name].initial_amount is not None:
                        LOGGER.debug("Found variable %s that has amount", _name)
                        is_amount = True
                        break
        if is_amount:
            fn = fn * sympy.Symbol(species.compartment)  # type: ignore
            return fn, free_symbols(fn)

    # Fall back to assuming it is a concentration
    # Probably not a good guess, but what are you gonna do at this point :)
    return fn, free_symbols(fn)


def _convert_rxn_args_to_concs(
    fn: sympy.Expr, args: list[str], pmodel: pdata.Model
) -> tuple[sympy.Expr, list[str]]:
    replacements: dict[sympy.Symbol, sympy.Expr] = {}
    for arg in args:
        # the parsed species part is important to not
        # introduce conversion on things that aren't species
        if (species := pmodel.variables.get(arg)) is None:
            continue

        if species.is_concentration() and compartment_is_valid(pmodel, species):
            # We cancel out compartment instead of species / compartment due to
            # multi-species rxns. Otherwise we divide too often there
            compartment = sympy.Symbol(species.compartment)
            replacements[compartment] = compartment / compartment  # type: ignore
            # replacements[sympy.Symbol(arg)] = sympy.Symbol(f"{arg}_amount")

        elif (
            (species.initial_amount is not None or arg in pmodel.initial_assignments)
            and compartment_is_valid(pmodel, species)
            and not species.has_only_substance_units
        ):
            # Species is given in amounts of substance, but is represented in
            # concentration units when they appear in expressions

            old = sympy.Symbol(arg)
            compartment = sympy.Symbol(species.compartment)
            replacements[old] = old / compartment  # type: ignore

        elif species.has_only_substance_units:
            continue

    fn = cast(sympy.Expr, fn.subs(replacements))
    return fn, free_symbols(fn)


def _convert_stoich_tuple(x: tuple[float, str]) -> sympy.Expr:
    factor, name = x
    return sympy.Mul(sympy.Float(factor), sympy.Symbol(name))


def _convert_rxn_stoichs(
    rxn: pdata.Reaction, pmodel: pdata.Model
) -> dict[str, sympy.Float | str | data.Derived]:
    stoichiometry = {}

    for k, factor in rxn.stoichiometry.items():
        species = pmodel.variables[k]

        if (
            species.is_concentration()
            and compartment_is_valid(pmodel=pmodel, species=species)
            and not species.has_only_substance_units
        ):
            comp = sympy.Symbol(species.compartment)
            x = (
                _convert_stoich_tuple(factor)
                if isinstance(factor, tuple)
                else sympy.Float(factor)
            )
            fn = x * comp  # type: ignore
            stoichiometry[f"{k}_amount"] = data.Derived(
                fn=fn,
                args=free_symbols(fn),
            )
        else:
            stoichiometry[k] = (
                data.Derived(fn=_convert_stoich_tuple(factor), args=[factor[1]])
                if isinstance(factor, tuple)
                else sympy.Float(factor)
            )
    return stoichiometry


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
            tmodel.parameters[k] = data.Parameter(
                value=sympy.Float(par.value), unit=None
            )
        else:
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(par.value), unit=None
            )


def _div_expr(x: str | sympy.Symbol, y: str | sympy.Symbol) -> sympy.Expr:
    x = sympy.Symbol(x) if isinstance(x, str) else x
    y = sympy.Symbol(y) if isinstance(y, str) else x
    return x / y  # type: ignore


def _mul_expr(x: str | sympy.Symbol, y: str | sympy.Symbol) -> sympy.Expr:
    x = sympy.Symbol(x) if isinstance(x, str) else x
    y = sympy.Symbol(y) if isinstance(y, str) else x
    return x * y  # type: ignore


def convert_compartments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for k, par in pmodel.compartments.items():
        if par.is_constant:
            tmodel.parameters[k] = data.Parameter(
                value=sympy.Float(par.size), unit=None
            )
        else:
            tmodel.variables[k] = data.Variable(
                initial_value=sympy.Float(par.size), unit=None
            )


def convert_functions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, fn in pmodel.functions.items():
        tmodel.functions[name] = data.Function(
            body=convert_mathml(fn.body, fns=tmodel.functions),
            args=[i.name for i in fn.args],
        )


def convert_rules(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, rr in pmodel.rate_rules.items():
        # Rate rules can create variables by SBML spec. Not cool
        if name not in tmodel.variables:
            tmodel.variables[name] = data.Variable(
                initial_value=sympy.Float(0.0), unit=None
            )

        stoichiometry = _convert_rate_rule_arg_to_conc(name, pmodel)
        tmodel.reactions[f"d{name}"] = data.Reaction(
            fn=convert_mathml(rr.body, fns=tmodel.functions),
            args=[i.name for i in rr.args],
            stoichiometry=stoichiometry,
        )

    for _ in pmodel.algebraic_rules.items():
        msg = "Algebraic rules not yet supported"
        raise NotImplementedError(msg)

    for name, ar in pmodel.assignment_rules.items():
        fn = convert_mathml(ar.body, fns=tmodel.functions)
        args = free_symbols(fn)

        for arg in args:
            if (var := pmodel.variables.get(arg)) is not None:
                if (
                    var.initial_concentration is not None
                    and compartment_is_valid(pmodel, var)
                    and not var.has_only_substance_units
                    and var.has_boundary_condition
                ):
                    fn = cast(sympy.Expr, fn.subs(arg, f"{arg}_amount"))
                elif (
                    var.initial_amount is not None
                    and compartment_is_valid(pmodel, var)
                    and not var.has_only_substance_units
                    and not var.has_boundary_condition
                    and not pmodel.compartments[
                        c := cast(str, var.compartment)
                    ].is_constant
                ):
                    subs = _div_expr(arg, c)
                    fn = cast(sympy.Expr, fn.subs(arg, subs))

        tmodel.derived[name] = data.Derived(
            fn=fn,
            args=free_symbols(fn),
        )


def convert_reactions(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, rxn in pmodel.reactions.items():
        fn = convert_mathml(rxn.body, fns=tmodel.functions)
        fn, args = _convert_rxn_args_to_concs(
            fn=fn,
            args=[i.name for i in rxn.args],
            pmodel=pmodel,
        )
        stoichiometry: dict[str, sympy.Float | str | data.Derived] = (
            _convert_rxn_stoichs(rxn=rxn, pmodel=pmodel)
        )
        pars_to_replace = {pn: f"{name}_{pn}" for pn in rxn.local_pars}
        fn = cast(sympy.Expr, fn.subs(pars_to_replace))

        tmodel.reactions[name] = data.Reaction(
            fn=fn,
            args=[pars_to_replace.get(i, i) for i in args],
            stoichiometry=stoichiometry,
        )

        for pn, par in rxn.local_pars.items():
            tmodel.parameters[pars_to_replace[pn]] = data.Parameter(
                value=sympy.Float(par.value), unit=None
            )


def convert_initial_assignments(pmodel: pdata.Model, tmodel: data.Model) -> None:
    for name, ia in pmodel.initial_assignments.items():
        # Assign normal parameter
        if (pp := pmodel.parameters.get(name)) is not None and pp.is_constant:
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue

        # Assign constant variable
        if pmodel.variables.get(name) is not None and variable_is_constant(
            name, pmodel
        ):
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue

        # Assign constant compartment
        if (el := pmodel.compartments.get(name)) is not None and el.is_constant:
            tmodel.initial_assignments[name] = data.Derived(
                fn=convert_mathml(ia.body, fns=tmodel.functions),
                args=[i.name for i in ia.args],
            )
            continue

        # Otherwise it's a normal variable
        fn, args = _convert_init_target_to_conc(
            name=name,
            fn=convert_mathml(ia.body, fns=tmodel.functions),
            args=[i.name for i in ia.args],
            pmodel=pmodel,
        )
        derived = data.Derived(fn=fn, args=args)
        variable = tmodel.variables.get(name)
        if variable is None:
            tmodel.variables[name] = data.Variable(
                initial_value=sympy.Float(0.0), unit=None
            )

        tmodel.initial_assignments[name] = derived


def remove_duplicate_entries(tmodel: data.Model) -> None:
    for name in tmodel.derived:
        if name in tmodel.parameters:
            del tmodel.parameters[name]
        elif name in tmodel.variables:
            del tmodel.variables[name]


def transform(doc: pdata.Document) -> data.Model:
    for plugin in doc.plugins:
        if plugin.name == "comp":
            msg = "Comp package not yet supported."
            raise NotImplementedError(msg)

    pmodel = doc.model
    if pmodel.conversion_factor is not None:
        msg = "Conversion factors not yet supported"
        raise NotImplementedError(msg)

    tmodel = data.Model(name=pmodel.name)  # type: ignore
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
