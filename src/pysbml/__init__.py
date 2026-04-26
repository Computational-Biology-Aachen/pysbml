"""SBML reader for ODE models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pysbml.parse import load_document

from . import parse, transform

if TYPE_CHECKING:
    from pathlib import Path


def load_model(file: str | Path) -> parse.data.Model:
    """Load and parse an SBML file, returning the raw parsed model."""
    return load_document(file).model


def load_and_transform_model(file: str | Path) -> transform.data.Model:
    """Load, parse and transform an SBML file into a simplified model."""
    return transform.transform(load_document(file))
