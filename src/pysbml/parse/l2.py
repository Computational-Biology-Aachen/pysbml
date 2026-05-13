"""SBML Level 2 parser: extends l1 with stoichiometryMath support."""

import logging
from collections import defaultdict

import libsbml

import pysbml.parse.data as pdata
from pysbml.parse import l1
from pysbml.parse.data import Derived, Model, Parameter, Reaction
from pysbml.parse.mathml import parse_sbml_math
from pysbml.parse.name_conversion import name_to_py
from pysbml.parse.units import get_unit_conversion

__all__ = ["LOGGER", "UNIT_CONVERSION", "parse", "parse_reactions"]

UNIT_CONVERSION = get_unit_conversion()

LOGGER = logging.getLogger(__name__)


def _parse_stoichiometries_l2(
    model: Model, reaction: libsbml.Reaction
) -> dict[str, int | list[tuple[float, str]]]:
    """Parse stoichiometries for L2, handling stoichiometryMath elements."""
    dynamic_stoichiometry: dict[str, list[tuple[float, str]]] = {}
    parsed_reactants: defaultdict[str, int] = defaultdict(int)

    reaction_id = name_to_py(reaction.getId())

    substrate: libsbml.SpeciesReference
    for i, substrate in enumerate(reaction.getListOfReactants()):
        species = name_to_py(substrate.getSpecies())

        if (ref := substrate.getId()) != "":
            model.parameters[ref] = Parameter(
                value=substrate.getStoichiometry(), is_constant=False, unit=None
            )
            if substrate.isSetStoichiometryMath():
                node = substrate.getStoichiometryMath().getMath()
                body, args = parse_sbml_math(node=node)
                model.assignment_rules[ref] = Derived(body=body, args=args)
            dynamic_stoichiometry.setdefault(species, []).append((-1.0, ref))

        elif substrate.isSetStoichiometryMath():
            node = substrate.getStoichiometryMath().getMath()
            body, args = parse_sbml_math(node=node)
            synth = f"_stoich_{reaction_id}_{species}_r{i}"
            model.parameters[synth] = Parameter(value=1.0, is_constant=False, unit=None)
            model.assignment_rules[synth] = Derived(body=body, args=args)
            dynamic_stoichiometry.setdefault(species, []).append((-1.0, synth))

        elif species not in model.boundary_species:
            factor = substrate.getStoichiometry()
            if str(factor) == "nan":
                msg = f"Cannot parse stoichiometry: {factor} for reaction {reaction.getId()}"
                raise ValueError(msg)
            parsed_reactants[species] -= factor

    product: libsbml.SpeciesReference
    parsed_products: defaultdict[str, int] = defaultdict(int)
    for i, product in enumerate(reaction.getListOfProducts()):
        species = name_to_py(product.getSpecies())
        if (ref := product.getId()) != "":
            model.parameters[ref] = Parameter(
                value=product.getStoichiometry(), is_constant=False, unit=None
            )
            if product.isSetStoichiometryMath():
                node = product.getStoichiometryMath().getMath()
                body, args = parse_sbml_math(node=node)
                model.assignment_rules[ref] = Derived(body=body, args=args)
            dynamic_stoichiometry.setdefault(species, []).append((1.0, ref))

        elif product.isSetStoichiometryMath():
            node = product.getStoichiometryMath().getMath()
            body, args = parse_sbml_math(node=node)
            synth = f"_stoich_{reaction_id}_{species}_p{i}"
            model.parameters[synth] = Parameter(value=1.0, is_constant=False, unit=None)
            model.assignment_rules[synth] = Derived(body=body, args=args)
            dynamic_stoichiometry.setdefault(species, []).append((1.0, synth))

        elif species not in model.boundary_species:
            factor = product.getStoichiometry()
            if str(factor) == "nan":
                msg = f"Cannot parse stoichiometry: {factor} for reaction {reaction.getId()}"
                raise ValueError(msg)
            parsed_products[species] += factor

    stoichiometries = dict(parsed_reactants)
    for species, value in parsed_products.items():
        stoichiometries[species] = stoichiometries.get(species, 0) + value

    return stoichiometries | dynamic_stoichiometry


def parse_reactions(model: Model, sbml_model: libsbml.Model) -> None:
    """Parse reactions, using L2 stoichiometry parser that handles stoichiometryMath."""
    for reaction in sbml_model.getListOfReactions():
        name = name_to_py(reaction.getId())
        kinetic_law = reaction.getKineticLaw()

        if kinetic_law is None:
            LOGGER.warning("Reaction %s has no kinetic law", name)
            continue

        node = kinetic_law.getMath()
        body, args = parse_sbml_math(node=node)

        model.reactions[name] = Reaction(
            body=body,
            args=args,
            stoichiometry=_parse_stoichiometries_l2(model=model, reaction=reaction),
            local_pars=l1.parse_local_parameters(kinetic_law=kinetic_law),
            fast=reaction.getFast(),
        )


def parse(lib_model: libsbml.Model, level: int) -> Model:  # noqa: ARG001
    model = pdata.Model(
        name=lib_model.getName(),
        conversion_factor=None
        if (conv := lib_model.getConversionFactor()) == ""
        else conv,
    )
    l1.parse_units(model=model, lib_model=lib_model)
    l1.parse_constraints(model=model, lib_model=lib_model)
    l1.parse_events(model=model, lib_model=lib_model)
    l1.parse_compartments(model=model, lib_model=lib_model)
    l1.parse_parameters(model=model, lib_model=lib_model)
    l1.parse_species(model=model, lib_model=lib_model)
    l1.parse_functions(model=model, lib_model=lib_model)
    l1.parse_initial_assignments(model=model, lib_model=lib_model)
    l1.parse_rules(model=model, sbml_model=lib_model)
    parse_reactions(model=model, sbml_model=lib_model)
    return model
