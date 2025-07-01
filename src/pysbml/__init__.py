"""SBML reader for ODE models"""

from __future__ import annotations

from pathlib import Path

from pysbml.parse import load_document

from . import parse, transform


def load_model(file: str | Path) -> parse.data.Model:
    return load_document(file).model


def load_and_transform_model(file: str | Path) -> transform.data.Model:
    return transform.transform(load_document(file).model)
