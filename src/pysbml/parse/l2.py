import logging

import libsbml

from pysbml.parse.data import Model
from pysbml.parse.l1 import (
    parse_compartments,
    parse_constraints,
    parse_events,
    parse_functions,
    parse_initial_assignments,
    parse_parameters,
    parse_reactions,
    parse_rules,
    parse_units,
    parse_variables,
)
from pysbml.parse.units import get_unit_conversion

UNIT_CONVERSION = get_unit_conversion()

LOGGER = logging.getLogger(__name__)


def nan_to_zero(value: float) -> float:
    return 0 if str(value) == "nan" else value


def parse(lib_model: libsbml.Model, level: int) -> Model:
    """Parse sbml model."""

    model = Model(
        name=lib_model.getName(),  # type: ignore
    )
    parse_units(model, lib_model=lib_model)
    parse_constraints(model, lib_model=lib_model)
    parse_events(model, lib_model=lib_model)
    parse_units(model, lib_model=lib_model)
    parse_compartments(model, lib_model=lib_model)
    parse_parameters(model, lib_model=lib_model)
    parse_variables(model, lib_model=lib_model)
    parse_functions(model, lib_model=lib_model)
    parse_initial_assignments(model, lib_model=lib_model)
    parse_rules(model, sbml_model=lib_model)
    parse_reactions(model, sbml_model=lib_model)
    return model
