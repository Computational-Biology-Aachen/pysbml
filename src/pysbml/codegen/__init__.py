from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, cast

import sympy
from sympy.printing.pycode import pycode

if TYPE_CHECKING:
    from pysbml.transform import data as tdata

INDENT = "    "


@dataclass
class Dependency:
    """Container class for building dependency tree."""

    name: str
    required: set[str]


def _sort_dependencies(
    available: set[str],
    elements: list[Dependency],
) -> list[str]:
    """Sort model elements topologically based on their dependencies.

    Args:
        available: Set of available component names
        elements: List of (name, dependencies, supplier) tuples to sort

    Returns:
        List of element names in dependency order

    Raises:
        SortError: If circular dependencies are detected

    """

    order = []
    # FIXME: what is the worst case here?
    max_iterations = len(elements) ** 2
    queue: SimpleQueue[Dependency] = SimpleQueue()
    for dependency in elements:
        queue.put(dependency)

    last_name = None
    i = 0
    while True:
        try:
            dependency = queue.get_nowait()
        except Empty:
            break
        if dependency.required.issubset(available):
            available.add(dependency.name)
            order.append(dependency.name)

        else:
            if last_name == dependency.name:
                order.append(last_name)
                break
            queue.put(dependency)
            last_name = dependency.name
        i += 1

        # Failure case
        if i > max_iterations:
            unsorted = []
            while True:
                try:
                    unsorted.append(queue.get_nowait().name)
                except Empty:
                    break

            msg = "Fuck"
            raise TypeError(msg)
    return order


def free_symbols(expr: sympy.Expr) -> list[str]:
    return [i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol)]


def codegen_expr(expr: sympy.Expr) -> str:
    return cast(str, pycode(expr, fully_qualified_modules=True)).replace(
        "math.factorial", "scipy.special.factorial"
    )


def codegen_value(val: sympy.Float) -> str:
    return cast(str, pycode(val, fully_qualified_modules=True))


def _to_sympy_types(
    x: str | float | tdata.Expr,
) -> tdata.Expr:
    if isinstance(x, str):
        return sympy.Symbol(x)
    if isinstance(x, float | int):
        return sympy.Float(x)
    return x


def _mul_expr(
    x: str | float | tdata.Expr,
    y: str | float | tdata.Expr,
) -> sympy.Expr:
    return _to_sympy_types(x) * _to_sympy_types(y)  # type: ignore


def codegen(model: tdata.Model) -> str:
    # Do calculations
    variable_names = sorted(model.variables)

    exprs = dict(**model.derived, **{k: rxn.expr for k, rxn in model.reactions.items()})
    init_exprs = dict(exprs, **model.initial_assignments)

    initial_order = _sort_dependencies(
        available=(set(model.parameters) | set(model.variables) | {"time"})
        ^ set(model.initial_assignments),
        elements=[
            Dependency(name=name, required=set(free_symbols(expr)))
            for name, expr in init_exprs.items()
        ],
    )

    order = _sort_dependencies(
        available=set(model.parameters) | set(model.variables) | {"time"},
        elements=[
            Dependency(name=name, required=set(free_symbols(expr)))
            for name, expr in exprs.items()
        ],
    )

    diff_eqs: dict[str, sympy.Expr] = {}
    for name, rxn in model.reactions.items():
        for var, stoich in rxn.stoichiometry.items():
            diff_eqs[var] = sympy.Add(
                diff_eqs.get(var, sympy.Float(0.0)), _mul_expr(stoich, name)
            )

    all_args = set()
    for derived in model.derived.values():
        all_args.update(free_symbols(derived))
    for rxn in model.reactions.values():
        all_args.update(free_symbols(rxn.expr))
        all_args.update(rxn.stoichiometry)
    for diff_eq in diff_eqs.values():
        all_args.update(free_symbols(diff_eq))

    # Not all variables always have an equation, because fuck why
    variable_names = [i for i in variable_names if i in diff_eqs]

    # Actual codegen
    source = [
        "import math",
        "import scipy.special",
        "import pandas as pd",
        "",
        "time: float = 0.0",
    ]

    for name, par in model.parameters.items():
        if name in model.initial_assignments:
            continue
        source.append(f"{name}: float = {codegen_value(par.value)}")

    for name, var in model.variables.items():
        if name in model.initial_assignments:
            continue
        source.append(f"{name}: float = {codegen_value(var.value)}")

    # Initial assignments
    if len(initial_order) > 0:
        source.append("\n# Initial assignments")
    source.extend(
        f"{name} = {codegen_expr(init_exprs[name])}" for name in initial_order
    )

    # Write y0
    source.append(f"y0 = [{', '.join(name for name in variable_names)}]")
    source.append(f"variable_names = {variable_names}")

    # Write main function
    source.append(
        "\ndef model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:",
    )
    if len(variable_names) > 0:
        source.append(
            f"{INDENT}{', '.join(variable_names)} = variables"
            if len(variable_names) > 1
            else f"{INDENT}{variable_names[0]}, = variables"
        )

    for name in order:
        if name not in all_args:
            continue
        source.append(f"{INDENT}{name}: float = {codegen_expr(exprs[name])}")

    for name, eq in diff_eqs.items():
        source.append(
            f"{INDENT}d{name}dt: float = {codegen_expr(cast(sympy.Expr, eq.subs(1.0, 1)))}"
        )

    returns = (f"d{i}dt" for i in variable_names)
    if len(variable_names) == 1:
        source.append(f"{INDENT}return ({next(iter(returns))},)")
    else:
        source.append(f"{INDENT}return {', '.join(returns)}")

    # Additional variables
    source.append(
        "\n\ndef derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:"
    )
    if len(variable_names) > 0:
        source.append(
            f"{INDENT}{', '.join(variable_names)} = variables"
            if len(variable_names) > 1
            else f"{INDENT}{variable_names[0]}, = variables"
        )

    source.extend(
        f"{INDENT}{name}: float = {codegen_expr(exprs[name])}" for name in order
    )

    source.extend(
        [
            f"{INDENT}return {{",
            *(f"{INDENT * 2}{name!r}: {name}," for name in model.parameters),
            *(f"{INDENT * 2}{name!r}: {name}," for name in order),
            f"{INDENT}}}",
        ]
    )

    return "\n".join(source)
