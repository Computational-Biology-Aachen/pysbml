"""Data classes for the transformed (sympy-based) SBML model representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import wadler_lindig as wl

from pysbml.parse.data import md_table_from_dict

__all__ = [
    "Event",
    "Expr",
    "Model",
    "Parameter",
    "Reaction",
    "Stoichiometry",
    "Variable",
]

if TYPE_CHECKING:
    import sympy
    from sympy.physics.units.quantities import Quantity


def _md_eq(s: Expr) -> str:
    return f"${s}$".replace("|", r"\|").replace("&", r"\&").replace("_", r"\_")


type Expr = sympy.Symbol | sympy.Float | sympy.Expr
type Stoichiometry = dict[str, Expr]


@dataclass(kw_only=True, slots=True)
class Event:
    """A transformed SBML event with sympy trigger and assignment expressions."""

    trigger: sympy.Expr | None
    assignments: dict[str, sympy.Expr]
    initial_value: bool
    persistent: bool
    delay: sympy.Expr | None
    priority: sympy.Expr | None
    use_values_from_trigger_time: bool

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Parameter:
    """A constant model parameter with a sympy float value and optional unit."""

    value: sympy.Float
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Variable:
    """A dynamic model variable (species) with a sympy float initial value and optional unit."""

    value: sympy.Float
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Reaction:
    """A transformed reaction with a sympy kinetic law expression and stoichiometry map."""

    expr: sympy.Expr
    stoichiometry: Stoichiometry

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Model:
    """The complete transformed SBML model with sympy expressions for all components."""

    name: str
    units: dict[str, Quantity] = field(default_factory=dict)
    functions: dict[str, Expr] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    variables: dict[str, Variable] = field(default_factory=dict)
    derived: dict[str, Expr] = field(default_factory=dict)
    reactions: dict[str, Reaction] = field(default_factory=dict)
    initial_assignments: dict[str, Expr] = field(default_factory=dict)
    events: dict[str, Event] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)

    def _repr_markdown_(self) -> str:
        content = [f"# {self.name}"]

        if len(self.functions) > 0:
            content.append("# Functions")
            content.append(
                md_table_from_dict(
                    headers=[
                        "name",
                        "body",
                    ],
                    els=[(k, _md_eq(v)) for k, v in self.functions.items()],
                )
            )

        if len(self.parameters) > 0:
            content.append("# Parameters")
            content.append(
                md_table_from_dict(
                    headers=[
                        "name",
                        "value",
                        "unit",
                    ],
                    els=[(k, v.value, v.unit) for k, v in self.parameters.items()],
                )
            )
        if len(self.variables) > 0:
            content.append("# Variables")
            content.append(
                md_table_from_dict(
                    headers=[
                        "name",
                        "value",
                        "unit",
                    ],
                    els=[
                        (
                            k,
                            _md_eq(v.value),
                            v.unit,
                        )
                        for k, v in self.variables.items()
                    ],
                )
            )
        if len(self.derived) > 0:
            content.append("# Derived")
            content.append(
                md_table_from_dict(
                    headers=["name", "fn"],
                    els=[(k, _md_eq(v)) for k, v in self.derived.items()],
                )
            )
        if len(self.initial_assignments) > 0:
            content.append("# Initial assignments")
            content.append(
                md_table_from_dict(
                    headers=["name", "fn"],
                    els=[(k, _md_eq(v)) for k, v in self.initial_assignments.items()],
                )
            )

        if len(self.reactions) > 0:
            content.append("# Reactions")
            content.append(
                md_table_from_dict(
                    headers=["name", "fn", "stoichiometry"],
                    els=[
                        (k, _md_eq(v.expr), v.stoichiometry)
                        for k, v in self.reactions.items()
                    ],
                )
            )
        return "\n".join(content)
