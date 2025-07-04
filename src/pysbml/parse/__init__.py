from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import libsbml

from . import data, l1, l2, l3

if TYPE_CHECKING:
    from .data import Model

__all__ = [
    "data",
    "parse",
]


def parse(lib_model: libsbml.Model, version: int, level: int) -> Model:
    """Parse sbml model."""
    if version != 2 or level != 3:
        msg = f"Version {version}, level {level} unsupported"
        raise NotImplementedError(msg)

    match version:
        case 1:
            return l1.parse(lib_model=lib_model, level=level)
        case 2:
            return l2.parse(lib_model=lib_model, level=level)
        case 3:
            return l3.parse(lib_model=lib_model, level=level)
        case _:
            msg = f"Unknown SBML version {version}"
            raise NotImplementedError(msg)


def load_document(file: str | Path) -> data.Document:
    if not Path(file).exists():
        msg = "Model file does not exist"
        raise OSError(msg)

    doc = libsbml.readSBMLFromFile(str(file))
    version = cast(int, doc.version)
    level = cast(int, doc.level)

    model: libsbml.Model = doc.getModel()
    if len(model.getListOfAllElementsFromPlugins()):
        msg = "Plugin handling not yet implemented"
        raise NotImplementedError(msg)

    return data.Document(
        model=parse(model, version=version, level=level),
        plugins=[
            data.Plugin(name=(doc.getPlugin(i).package_name))
            for i in range(doc.num_plugins)
        ],
    )
