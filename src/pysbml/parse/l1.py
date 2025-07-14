import itertools as it
import logging
import math
from collections import defaultdict

import libsbml

import pysbml.parse.data as pdata
from pysbml.parse.data import (
    Assignment,
    AtomicUnit,
    Compartment,
    CompositeUnit,
    Constraint,
    Delay,
    Derived,
    Event,
    Function,
    Model,
    Parameter,
    Priority,
    Reaction,
    Species,
    Trigger,
)
from pysbml.parse.mathml import parse_sbml_math
from pysbml.parse.name_conversion import name_to_py
from pysbml.parse.units import get_unit_conversion

__all__ = [
    "LOGGER",
    "UNIT_CONVERSION",
    "handle_nan",
    "parse",
    "parse_compartments",
    "parse_constraints",
    "parse_events",
    "parse_functions",
    "parse_initial_assignments",
    "parse_parameters",
    "parse_reactions",
    "parse_rules",
    "parse_species",
    "parse_units",
]

UNIT_CONVERSION = get_unit_conversion()

LOGGER = logging.getLogger(__name__)


def handle_nan(value: float) -> float:
    return math.nan if str(value) == "nan" else value


def parse_constraints(model: Model, lib_model: libsbml.Model) -> None:
    for con in lib_model.getListOfConstraints():
        name = con.getId()

        math, args = None, []
        if (node := con.getMath()) is not None:
            math, args = parse_sbml_math(node)

        message = con.getMessage()
        model.constraints[name] = Constraint(math=math, args=args, message=message)


def _parse_event_trigger(trigger: libsbml.Trigger) -> Trigger | None:
    if trigger is None:
        return None

    math, args = None, []
    if (lib_math := trigger.getMath()) is not None:
        math, args = parse_sbml_math(lib_math)

    return Trigger(
        math=math,
        args=args,
        initial_value=trigger.getInitialValue(),
        persistent=trigger.getPersistent(),
    )


def _parse_event_delay(delay: libsbml.Delay | None) -> Delay | None:
    if delay is None:
        return None

    math, args = None, []
    if (lib_math := delay.getMath()) is not None:
        math, args = parse_sbml_math(lib_math)

    return Delay(math=math, args=args)


def _parse_event_priority(priority: libsbml.Priority) -> Priority | None:
    if priority is None:
        return None

    math, args = None, []
    if (lib_math := priority.getMath()) is not None:
        math, args = parse_sbml_math(lib_math)
    return Priority(math, args)


def _parse_event_assignment(assignment: libsbml.AssignmentRule) -> Assignment:
    variable = assignment.getVariable()
    if (lib_math := assignment.getMath()) is None:
        return Assignment(variable, None, [])

    math, args = parse_sbml_math(lib_math)
    return Assignment(variable, math, args)


def parse_events(model: Model, lib_model: libsbml.Model) -> None:
    for e in lib_model.getListOfEvents():
        name = e.getId()

        model.events[name] = Event(
            trigger=_parse_event_trigger(e.getTrigger()),
            delay=_parse_event_delay(e.getDelay()),
            priority=_parse_event_priority(e.getPriority()),
            assignments=[
                _parse_event_assignment(e.getEventAssignment(i))
                for i in range(e.getNumEventAssignments())
            ],
        )


def parse_units(model: Model, lib_model: libsbml.Model) -> None:
    unit_definition: libsbml.UnitDefinition
    unit: libsbml.Unit

    for unit_definition in lib_model.getListOfUnitDefinitions():
        composite_id = unit_definition.getId()
        local_units = []
        for unit in unit_definition.getListOfUnits():
            atomic_unit = AtomicUnit(
                kind=UNIT_CONVERSION[unit.getKind()],
                scale=unit.getScale(),
                exponent=unit.getExponent(),
                multiplier=unit.getMultiplier(),
            )
            local_units.append(atomic_unit.kind)
            model.atomic_units[atomic_unit.kind] = atomic_unit

        model.composite_units[composite_id] = CompositeUnit(
            sbml_id=composite_id,
            units=local_units,
        )


def parse_compartments(model: Model, lib_model: libsbml.Model) -> None:
    compartment: libsbml.Compartment

    for compartment in lib_model.getListOfCompartments():
        sbml_id = name_to_py(compartment.getId())
        model.compartments[sbml_id] = Compartment(
            name=compartment.getName(),
            dimensions=compartment.getSpatialDimensions(),
            size=handle_nan(compartment.getSize()),
            units=compartment.getUnits(),
            is_constant=compartment.getConstant(),
        )


def parse_parameters(model: Model, lib_model: libsbml.Model) -> None:
    parameter: libsbml.Parameter

    for parameter in lib_model.getListOfParameters():
        model.parameters[name_to_py(parameter.getId())] = Parameter(
            value=handle_nan(parameter.getValue()),
            is_constant=parameter.getConstant(),
            unit=parameter.getUnits(),
        )


def parse_species(model: Model, lib_model: libsbml.Model) -> None:
    for compound in lib_model.getListOfSpecies():
        compound_id = name_to_py(compound.getId())
        conversion_factor: str | None = (
            cf if bool(cf := compound.getConversionFactor()) else None
        )
        initial_amount = (
            None if str(init := compound.getInitialAmount()) == "nan" else init
        )
        initial_concentration = (
            None if str(init := compound.getInitialConcentration()) == "nan" else init
        )
        has_boundary_condition: bool = compound.getBoundaryCondition()
        if has_boundary_condition:
            model.boundary_species.add(compound_id)

        model.variables[compound_id] = Species(
            compartment=comp if (comp := compound.getCompartment()) else None,
            conversion_factor=conversion_factor,
            initial_amount=initial_amount,
            initial_concentration=initial_concentration,
            substance_units=compound.getSubstanceUnits(),
            has_only_substance_units=compound.getHasOnlySubstanceUnits(),
            has_boundary_condition=has_boundary_condition,
            is_constant=compound.getConstant(),
        )


def parse_functions(model: Model, lib_model: libsbml.Model) -> None:
    for func in lib_model.getListOfFunctionDefinitions():
        # Sure, why not just have one name
        name = func.getName()
        sbml_id = func.getId()
        if sbml_id is None or sbml_id == "":
            sbml_id = name
        elif name is None or name == "":
            name = sbml_id
        name = name_to_py(name)

        if (node := func.getMath()) is None:
            msg = f"Function {name} has no math element"
            LOGGER.warning(msg)
            continue

        body, args = parse_sbml_math(node=node)
        model.functions[name] = Function(
            body=body,
            args=args,
        )


def parse_initial_assignments(model: Model, lib_model: libsbml.Model) -> None:
    for assignment in lib_model.getListOfInitialAssignments():
        name = name_to_py(assignment.getSymbol())

        if (node := assignment.getMath()) is None:
            msg = f"Initial assignment {name} has no math element"
            LOGGER.warning(msg)
            continue

        body, args = parse_sbml_math(node)
        model.initial_assignments[name] = Derived(
            body=body,
            args=args,
        )


def _parse_algebraic_rule(model: Model, rule: libsbml.AlgebraicRule) -> None:
    name: str = name_to_py(rule.getId())

    if (node := rule.getMath()) is None:
        msg = f"Algebraic rule {name} has no math element"
        LOGGER.warning(msg)
        return

    body, args = parse_sbml_math(node=node)

    model.algebraic_rules[name] = Derived(
        body=body,
        args=args,
    )


def _parse_assignment_rule(model: Model, rule: libsbml.AssignmentRule) -> None:
    name: str = name_to_py(rule.getId())

    if (node := rule.getMath()) is None:
        msg = f"Assignment rule {name} has no math element"
        LOGGER.warning(msg)
        return

    body, args = parse_sbml_math(node=node)

    model.assignment_rules[name] = Derived(
        body=body,
        args=args,
    )


def _parse_rate_rule(model: Model, rule: libsbml.RateRule) -> None:
    name: str = name_to_py(rule.getId())

    if (node := rule.getMath()) is None:
        msg = f"Rate rule {name} has no math element"
        LOGGER.warning(msg)
        return

    body, args = parse_sbml_math(node=node)

    model.rate_rules[name] = Derived(
        body=body,
        args=args,
    )


def parse_rules(model: Model, sbml_model: libsbml.Model) -> None:
    """Parse rules and separate them by type."""
    for rule in sbml_model.getListOfRules():
        if rule.element_name == "algebraicRule":
            _parse_algebraic_rule(model, rule=rule)
        elif rule.element_name == "assignmentRule":
            _parse_assignment_rule(model, rule=rule)
        elif rule.element_name == "rateRule":
            _parse_rate_rule(model, rule=rule)
        else:
            msg = "Unknown rate type"
            raise ValueError(msg)


def _parse_local_parameters(
    kinetic_law: libsbml.KineticLaw,
) -> dict[str, Parameter]:
    """Parse local parameters."""
    pars = {}

    # Because having one extra container of parameters isn't enough
    for parameter in it.chain(
        kinetic_law.getListOfLocalParameters(),
        kinetic_law.getListOfParameters(),
    ):
        name = name_to_py(parameter.getId())
        pars[name] = Parameter(
            value=handle_nan(parameter.getValue()),
            is_constant=parameter.getConstant(),
            unit=None,
        )
    return pars


def _parse_stoichiometries(
    model: Model, reaction: libsbml.Reaction
) -> dict[str, int | list[tuple[float, str]]]:
    """Parse reaction stoichiometries

    Stoichiometries can be multiple things
    - species
    - boundary species
    - references
    """
    dynamic_stoichiometry: dict[str, list[tuple[float, str]]] = {}
    parsed_reactants: defaultdict[str, int] = defaultdict(int)

    substrate: libsbml.SpeciesReference
    for substrate in reaction.getListOfReactants():
        species = name_to_py(substrate.getSpecies())

        # Only species references have Id set
        if (ref := substrate.getId()) != "":
            model.parameters[ref] = Parameter(
                value=substrate.getStoichiometry(), is_constant=False, unit=None
            )
            # Test 1434 has a case where there are multiple references to the same
            # species...
            dynamic_stoichiometry.setdefault(species, []).append((-1.0, ref))

        # Boundary species can safely be ignored
        elif species not in model.boundary_species:
            factor = substrate.getStoichiometry()
            if str(factor) == "nan":
                msg = f"Cannot parse stoichiometry: {factor} for reaction {reaction.getId()}"
                raise ValueError(msg)
            parsed_reactants[species] -= factor

    product: libsbml.SpeciesReference
    parsed_products: defaultdict[str, int] = defaultdict(int)
    for product in reaction.getListOfProducts():
        species = name_to_py(product.getSpecies())
        if (ref := product.getId()) != "":
            dynamic_stoichiometry.setdefault(species, []).append((1.0, ref))
            model.parameters[ref] = Parameter(
                value=product.getStoichiometry(), is_constant=False, unit=None
            )
        elif species not in model.boundary_species:
            factor = product.getStoichiometry()
            if str(factor) == "nan":
                msg = f"Cannot parse stoichiometry: {factor} for reaction {reaction.getId()}"
                raise ValueError(msg)
            parsed_products[species] += factor

    # Combine stoichiometries
    # Hint: you can't just combine the dictionaries, as you have cases like
    # S1 + S2 -> 2S2, which have to be combined to S1 -> S2
    stoichiometries = dict(parsed_reactants)
    for species, value in parsed_products.items():
        stoichiometries[species] = stoichiometries.get(species, 0) + value

    return stoichiometries | dynamic_stoichiometry


def parse_reactions(model: Model, sbml_model: libsbml.Model) -> None:
    for reaction in sbml_model.getListOfReactions():
        name = name_to_py(reaction.getId())
        kinetic_law = reaction.getKineticLaw()

        if kinetic_law is None:
            msg = f"Reaction {name} has no kinetic law"
            LOGGER.warning(msg)
            continue

        node = kinetic_law.getMath()
        body, args = parse_sbml_math(node=node)

        model.reactions[name] = Reaction(
            body=body,
            args=args,
            stoichiometry=_parse_stoichiometries(model=model, reaction=reaction),
            local_pars=_parse_local_parameters(
                kinetic_law=kinetic_law,
            ),
        )


def parse(
    lib_model: libsbml.Model,
    level: int,  # noqa: ARG001
) -> Model:
    """Parse sbml model."""

    model = pdata.Model(
        name=lib_model.getName(),  # type: ignore
        conversion_factor=None  # type: ignore
        if (conv := lib_model.getConversionFactor()) == ""
        else conv,
    )
    parse_units(model=model, lib_model=lib_model)
    parse_constraints(model=model, lib_model=lib_model)
    parse_events(model=model, lib_model=lib_model)
    parse_compartments(model=model, lib_model=lib_model)
    parse_parameters(model=model, lib_model=lib_model)
    parse_species(model=model, lib_model=lib_model)
    parse_functions(model=model, lib_model=lib_model)
    parse_initial_assignments(model=model, lib_model=lib_model)
    parse_rules(model=model, sbml_model=lib_model)
    parse_reactions(model=model, sbml_model=lib_model)
    return model
