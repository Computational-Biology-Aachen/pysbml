"""Data classes for the raw SBML parse layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import wadler_lindig as wl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pysbml.parse.mathml import Base, Symbol


__all__ = [
    "Assignment",
    "AtomicUnit",
    "Compartment",
    "CompositeUnit",
    "Constraint",
    "Delay",
    "Derived",
    "Document",
    "Event",
    "Function",
    "Model",
    "Parameter",
    "Plugin",
    "Priority",
    "Reaction",
    "Species",
    "Trigger",
    "md_table_from_dict",
]


def md_table_from_dict(
    headers: list[str],
    els: Iterable[Iterable[Any]],
) -> str:
    """Render a markdown table from a list of headers and rows."""
    cols = len(headers)
    fmt = "| {} | ".format(" | ".join("{}" for _ in range(cols)))

    top = fmt.format(*headers)
    div = fmt.format(*("---" for _ in range(cols)))
    return "\n".join((top, div, *(fmt.format(*k) for k in els)))


@dataclass
class Plugin:
    """An SBML package plugin attached to a document."""

    name: str

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass
class Document:
    """Top-level SBML document containing the model and active plugins."""

    model: Model
    plugins: list[Plugin]

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class AtomicUnit:
    """A single SBML unit definition with kind, scale, exponent, and multiplier."""

    kind: str
    exponent: int
    scale: int
    multiplier: float

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class CompositeUnit:
    """A composite SBML unit composed of one or more atomic units."""

    sbml_id: str
    units: list

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Parameter:
    """An SBML parameter with a numeric value, constancy flag, and optional unit."""

    value: float
    is_constant: bool
    unit: str | None

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Compartment:
    """An SBML compartment with spatial dimensions, size, and constancy."""

    name: str
    dimensions: int
    size: float
    units: str
    is_constant: bool

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Species:
    """An SBML species with amount/concentration, compartment, and boundary flags."""

    compartment: str | None
    conversion_factor: str | None
    initial_amount: float | None
    initial_concentration: float | None
    substance_units: str | None
    has_only_substance_units: bool
    has_boundary_condition: bool
    is_constant: bool

    def is_concentration(self) -> bool:
        """Return True if this species is defined as a concentration."""
        if self.initial_concentration is not None:
            return True
        if self.has_only_substance_units:
            return False
        return False

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Derived:
    """A derived quantity defined by a MathML body and its free-symbol arguments."""

    body: Base
    args: list[Symbol]

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Function:
    """An SBML function definition with a MathML body and formal arguments."""

    body: Base
    args: list[Symbol]

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass(kw_only=True, slots=True)
class Reaction:
    """An SBML reaction with kinetic law, stoichiometry, and local parameters."""

    body: Base
    stoichiometry: Mapping[str, float | list[tuple[float, str]]]
    args: list[Symbol]
    local_pars: dict[str, Parameter] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)


@dataclass
class Trigger:
    """An SBML event trigger with optional math, initial value, and persistence."""

    math: Base | None
    args: list[Symbol]
    initial_value: bool
    persistent: bool


@dataclass
class Delay:
    """An SBML event delay defined by optional MathML."""

    math: Base | None
    args: list[Symbol]


@dataclass
class Priority:
    """An SBML event priority defined by optional MathML."""

    math: Base | None
    args: list[Symbol]


@dataclass
class Assignment:
    """A single variable assignment within an SBML event."""

    variable: str
    math: Base | None
    args: list[Symbol]


@dataclass
class Event:
    """An SBML event with trigger, delay, priority, and assignment actions."""

    assignments: list[Assignment]
    trigger: Trigger | None
    delay: Delay | None
    priority: Priority | None
    use_values_from_trigger_time: bool = True


@dataclass
class Constraint:
    """An SBML constraint with optional MathML and a message string."""

    math: Base | None
    args: list[Symbol]
    message: str


@dataclass(kw_only=True, slots=True)
class Model:
    """The complete parsed SBML model containing all components."""

    name: str
    conversion_factor: str | None = None
    # Collections
    boundary_species: set[str] = field(default_factory=set)
    # Parsed stuff
    events: dict[str, Event] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    atomic_units: dict[str, AtomicUnit] = field(default_factory=dict)
    composite_units: dict[str, CompositeUnit] = field(default_factory=dict)
    compartments: dict[str, Compartment] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    variables: dict[str, Species] = field(default_factory=dict)
    assignment_rules: dict[str, Derived] = field(default_factory=dict)
    algebraic_rules: dict[str, Derived] = field(default_factory=dict)
    rate_rules: dict[str, Derived] = field(default_factory=dict)
    initial_assignments: dict[str, Derived] = field(default_factory=dict)
    functions: dict[str, Function] = field(default_factory=dict)
    reactions: dict[str, Reaction] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return wl.pformat(self)

    def _repr_markdown_(self) -> str:
        content = [f"# {self.name}"]
        if len(self.functions) > 0:
            content.append("# Functions")
            content.append(
                md_table_from_dict(
                    headers=["name", "args", "body"],
                    els=[(k, v.args, v.body) for k, v in self.functions.items()],
                )
            )
        if len(self.compartments) > 0:
            content.append("# Compartment")
            content.append(
                md_table_from_dict(
                    headers=["name", "size", "is_constant"],
                    els=[
                        (k, v.size, v.is_constant) for k, v in self.compartments.items()
                    ],
                )
            )
        if len(self.variables) > 0:
            content.append("# Variables")
            content.append(
                md_table_from_dict(
                    headers=[
                        "name",
                        "amount",
                        "conc",
                        "constant",
                        "substance_units",
                        "compartment",
                        "only_substance_units",
                        "boundary_condition",
                    ],
                    els=[
                        (
                            k,
                            v.initial_amount,
                            v.initial_concentration,
                            v.is_constant,
                            v.substance_units,
                            v.compartment,
                            v.has_only_substance_units,
                            v.has_boundary_condition,
                        )
                        for k, v in self.variables.items()
                    ],
                )
            )
        if len(self.parameters) > 0:
            content.append("# Parameters")
            content.append(
                md_table_from_dict(
                    headers=["name", "value", "is_constant", "unit"],
                    els=[
                        (k, v.value, v.is_constant, v.unit)
                        for k, v in self.parameters.items()
                    ],
                )
            )
        if len(self.assignment_rules) > 0:
            content.append("# Assignment rules")
            content.append(
                md_table_from_dict(
                    headers=["name", "args", "body"],
                    els=[(k, v.args, v.body) for k, v in self.assignment_rules.items()],
                )
            )
        if len(self.algebraic_rules) > 0:
            content.append("# Algebraic rules")
            content.append(
                md_table_from_dict(
                    headers=["name", "args", "body"],
                    els=[(k, v.args, v.body) for k, v in self.algebraic_rules.items()],
                )
            )
        if len(self.rate_rules) > 0:
            content.append("# Rate_rules")
            content.append(
                md_table_from_dict(
                    headers=["name", "args", "body"],
                    els=[(k, v.args, v.body) for k, v in self.rate_rules.items()],
                )
            )

        if len(self.reactions) > 0:
            content.append("# Reactions")
            content.append(
                md_table_from_dict(
                    headers=["name", "body", "args", "stoichiometry", "local pars"],
                    els=[
                        (k, v.body, v.args, v.stoichiometry, v.local_pars)
                        for k, v in self.reactions.items()
                    ],
                )
            )
        if len(self.initial_assignments) > 0:
            content.append("# Initial assignments")
            content.append(
                md_table_from_dict(
                    headers=["name", "args", "body"],
                    els=[
                        (k, v.args, v.body) for k, v in self.initial_assignments.items()
                    ],
                )
            )
        return "\n".join(content)
