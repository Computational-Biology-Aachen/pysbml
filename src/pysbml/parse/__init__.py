"""SBML document parsing: dispatch to version-specific parsers."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import libsbml

from . import data, l1

if TYPE_CHECKING:
    from collections.abc import Generator

    from .data import Model

__all__ = [
    "data",
    "parse",
]

_IGNORED_PLUGIN_PACKAGES = frozenset({"fbc", "l3v2extendedmath", "groups", "layout"})


def parse(
    lib_model: libsbml.Model,
    level: int,  # Literal[1, 2, 3],
    version: int,  # Literal[1, 2, 3, 4, 5],
) -> Model:
    """Parse sbml model."""
    # FIXME: actually use different parsers here
    match version:
        case 1:
            return l1.parse(lib_model=lib_model, level=level)
        case 2:
            return l1.parse(lib_model=lib_model, level=level)
        case 3:
            return l1.parse(lib_model=lib_model, level=level)
        case _:
            msg = f"Unknown SBML version {version}"
            raise NotImplementedError(msg)


@contextlib.contextmanager
def _provide_external_models(file: Path) -> Generator[None, Any, None]:
    """Create symlinks for external comp model files if needed.

    Some SBML files reference external models by a bare name (e.g. enzyme_model.xml)
    while the test suite ships them with a version suffix (enzyme_model-l3v2.xml).
    We create temporary symlinks in the same directory so libsbml can resolve them.
    """
    xml_text = file.read_text()
    sources = re.findall(r'comp:source="([^"]+)"', xml_text)
    # Determine the version suffix from the filename (e.g. "l3v2" from "foo-l3v2.xml")
    version_match = re.search(r"-(l\d+v\d+)\.xml$", file.name)
    preferred_suffix = version_match.group(1) if version_match else "l3v2"
    created: list[Path] = []
    try:
        for src in sources:
            target = file.parent / src
            if not target.exists():
                stem = Path(src).stem
                candidates = sorted(file.parent.glob(f"{stem}-*.xml"))
                # Prefer the candidate matching the main file's version
                preferred = [c for c in candidates if preferred_suffix in c.name]
                pick = preferred or candidates
                if pick:
                    target.symlink_to(pick[0].name)
                    created.append(target)
        yield
    finally:
        for p in created:
            with contextlib.suppress(OSError):
                p.unlink()


def load_document(file: str | Path) -> data.Document:
    """Read an SBML file and return a parsed Document."""
    file = Path(file).resolve()
    if not file.exists():
        msg = "Model file does not exist"
        raise OSError(msg)

    with _provide_external_models(file):
        doc = libsbml.readSBMLFromFile(str(file))

        # Flatten SBML comp package models before parsing
        plugin_names = {doc.getPlugin(i).package_name for i in range(doc.num_plugins)}
        if "comp" in plugin_names:
            props = libsbml.ConversionProperties()
            props.addOption("flatten comp", True)  # noqa: FBT003 ; no control over api
            props.addOption("leave_ports_intact", False)  # noqa: FBT003 ; no control over api
            doc.convert(props)

    version = cast(int, doc.version)
    level = cast(int, doc.level)

    model: libsbml.Model = doc.getModel()
    remaining_packages = {
        e.getPackageName() for e in model.getListOfAllElementsFromPlugins()
    } - _IGNORED_PLUGIN_PACKAGES
    if remaining_packages:
        msg = f"Plugin handling not yet implemented: {remaining_packages}"
        raise NotImplementedError(msg)

    return data.Document(
        model=parse(model, version=version, level=level),
        plugins=[
            data.Plugin(name=(doc.getPlugin(i).package_name))
            for i in range(doc.num_plugins)
        ],
    )
