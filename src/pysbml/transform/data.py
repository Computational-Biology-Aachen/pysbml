from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import wadler_lindig as wl

from pysbml.parse.data import md_table_from_dict

__all__ = ["Expr", "Model", "Parameter", "Reaction", "Stoichiometry", "Variable"]

if TYPE_CHECKING:
    import sympy
    from sympy.physics.units.quantities import Quantity


def _md_eq(s: Expr) -> str:
    return f"${s}$".replace("|", r"\|").replace("&", r"\&").replace("_", r"\_")


type Expr = sympy.Symbol | sympy.Float | sympy.Expr
type Stoichiometry = dict[str, Expr]


@dataclass(kw_only=True, slots=True)
class Parameter:
    value: sympy.Float
    unit: Quantity | None

    def __repr__(self) -> str:
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Variable:
    value: sympy.Float
    unit: Quantity | None

    def __repr__(self) -> str:
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Reaction:
    expr: sympy.Expr
    stoichiometry: Stoichiometry

    def __repr__(self) -> str:
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Model:
    name: str
    units: dict[str, Quantity] = field(default_factory=dict)
    functions: dict[str, Expr] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    variables: dict[str, Variable] = field(default_factory=dict)
    derived: dict[str, Expr] = field(default_factory=dict)
    reactions: dict[str, Reaction] = field(default_factory=dict)
    initial_assignments: dict[str, Expr] = field(default_factory=dict)

    def __repr__(self) -> str:
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
