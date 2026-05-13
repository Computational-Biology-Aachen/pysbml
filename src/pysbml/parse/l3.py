"""Stub for SBML Level 3 parser (delegates to l1)."""

import logging

import libsbml

from pysbml.parse import l1
from pysbml.parse.data import Model
from pysbml.parse.units import get_unit_conversion

__all__ = ["LOGGER", "UNIT_CONVERSION", "parse"]

UNIT_CONVERSION = get_unit_conversion()

LOGGER = logging.getLogger(__name__)


def parse(lib_model: libsbml.Model, level: int) -> Model:
    return l1.parse(lib_model=lib_model, level=level)
