"""Code generation from transformed SBML models to Python ODE functions."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, cast

import sympy
from sympy.printing.pycode import PythonCodePrinter, pycode

if TYPE_CHECKING:
    from pysbml.transform import data as tdata

INDENT = "    "


class _SBMLPrinter(PythonCodePrinter):
    def _print_acot(self, expr: sympy.Expr) -> str:
        arg = self._print(expr.args[0])
        return f"(math.atan2(math.copysign(1.0, ({arg})), abs(({arg}))))"


_printer = _SBMLPrinter({"fully_qualified_modules": True})


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
    """Return sorted list of free symbol names in a sympy expression."""
    return [i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol)]


def codegen_expr(expr: sympy.Expr) -> str:
    """Generate Python source for a sympy expression."""
    return _printer.doprint(expr).replace("math.factorial", "scipy.special.factorial")


def codegen_value(val: sympy.Float) -> str:
    """Generate Python source for a sympy float literal."""
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
    """Generate a Python module string for a transformed SBML model."""
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

    # Events: compute here so event-target variables are in diff_eqs before filter
    active_events = [
        (name, ev) for name, ev in model.events.items() if ev.trigger is not None
    ]

    # Event-assigned variables need dSdt=0 even without a reaction
    event_target_vars = {
        var
        for _, ev in active_events
        for var in ev.assignments
        if var in model.variables
    }
    for v in event_target_vars:
        if v not in diff_eqs:
            diff_eqs[v] = sympy.Float(0.0)

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
        "from typing import TYPE_CHECKING",
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
        "\ndef model(time: float, variables: Iterable[float]) -> Iterable[float]:",
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
        "\n\ndef derived(time: float, variables: Iterable[float]) -> dict[str, float]:"
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

    # Derived expressions needed in event functions (computed in topological order)
    derived_in_order = [name for name in order if name in model.derived]

    for event_name, event in active_events:
        unpack = (
            (
                f"{INDENT}{', '.join(variable_names)} = variables"
                if len(variable_names) > 1
                else f"{INDENT}{variable_names[0]}, = variables"
            )
            if variable_names
            else None
        )

        source.append(
            f"\n\ndef _event_{event_name}_trigger(time: float, variables: Iterable[float]) -> float:"
        )
        if unpack:
            source.append(unpack)
        for dname in derived_in_order:
            source.append(f"{INDENT}{dname}: float = {codegen_expr(exprs[dname])}")  # noqa: PERF401
        source.append(
            f"{INDENT}return float({codegen_expr(cast(sympy.Expr, event.trigger))}) - 0.5"
        )
        source.append(f"_event_{event_name}_trigger.terminal = True")
        source.append(f"_event_{event_name}_trigger.direction = 1")

        source.append(
            f"\n\ndef _event_{event_name}_assignments(time: float, variables: Iterable[float]) -> dict[str, float]:"
        )
        if unpack:
            source.append(unpack)
        for dname in derived_in_order:
            source.append(f"{INDENT}{dname}: float = {codegen_expr(exprs[dname])}")  # noqa: PERF401
        source.extend(
            [
                f"{INDENT}return {{",
                *(
                    f"{INDENT * 2}{var!r}: {codegen_expr(assign_expr)},"
                    for var, assign_expr in event.assignments.items()
                ),
                f"{INDENT}}}",
            ]
        )

        if event.delay is not None:
            source.append(
                f"\n\ndef _event_{event_name}_delay(time: float, variables: Iterable[float]) -> float:"
            )
            if unpack:
                source.append(unpack)
            for dname in derived_in_order:
                source.append(f"{INDENT}{dname}: float = {codegen_expr(exprs[dname])}")  # noqa: PERF401
            source.append(
                f"{INDENT}return float({codegen_expr(cast(sympy.Expr, event.delay))})"
            )

        if event.priority is not None:
            source.append(
                f"\n\ndef _event_{event_name}_priority(time: float, variables: Iterable[float]) -> float:"
            )
            if unpack:
                source.append(unpack)
            for dname in derived_in_order:
                source.append(f"{INDENT}{dname}: float = {codegen_expr(exprs[dname])}")  # noqa: PERF401
            source.append(
                f"{INDENT}return float({codegen_expr(cast(sympy.Expr, event.priority))})"
            )

    event_tuples = ", ".join(
        "({trigger}, {assign}, {iv}, {pers}, {delay}, {prio}, {uvftt})".format(
            trigger=f"_event_{name}_trigger",
            assign=f"_event_{name}_assignments",
            iv=event.initial_value,
            pers=event.persistent,
            delay=f"_event_{name}_delay" if event.delay is not None else "None",
            prio=f"_event_{name}_priority" if event.priority is not None else "None",
            uvftt=event.use_values_from_trigger_time,
        )
        for name, event in active_events
    )
    source.append(f"\nevents: list = [{event_tuples}]")

    return "\n".join(source)
