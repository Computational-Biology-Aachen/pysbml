from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import libsbml

from .name_conversion import name_to_py
from .units import get_ast_types

LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from libsbml import ASTNode


@dataclass
class Base: ...


@dataclass
class Symbol(Base):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class Constant(Base):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class Boolean(Base):
    value: bool


@dataclass
class Integer(Base):
    value: int

    def __repr__(self) -> str:
        return f"{self.value}"


@dataclass
class Float(Base):
    value: float

    def __repr__(self) -> str:
        return f"{self.value:.2g}"


###############################################################################
# Unary fns
###############################################################################


@dataclass
class Abs(Base):
    child: Base


@dataclass
class Ceiling(Base):
    child: Base


@dataclass
class Exp(Base):
    child: Base


@dataclass
class Factorial(Base):
    child: Base


@dataclass
class Floor(Base):
    child: Base


@dataclass
class Ln(Base):
    child: Base


@dataclass
class Log(Base):
    base: Base
    child: Base


@dataclass
class Sqrt(Base):
    base: Base
    child: Base


@dataclass
class Sin(Base):
    child: Base


@dataclass
class Cos(Base):
    child: Base


@dataclass
class Tan(Base):
    child: Base


@dataclass
class Sec(Base):
    child: Base


@dataclass
class Csc(Base):
    child: Base


@dataclass
class Cot(Base):
    child: Base


@dataclass
class Asin(Base):
    child: Base


@dataclass
class Acos(Base):
    child: Base


@dataclass
class Atan(Base):
    child: Base


@dataclass
class Acot(Base):
    child: Base


@dataclass
class ArcSec(Base):
    child: Base


@dataclass
class ArcCsc(Base):
    child: Base


@dataclass
class Sinh(Base):
    child: Base


@dataclass
class Cosh(Base):
    child: Base


@dataclass
class Tanh(Base):
    child: Base


@dataclass
class Sech(Base):
    child: Base


@dataclass
class Csch(Base):
    child: Base


@dataclass
class Coth(Base):
    child: Base


@dataclass
class ArcSinh(Base):
    child: Base


@dataclass
class ArcCosh(Base):
    child: Base


@dataclass
class ArcTanh(Base):
    child: Base


@dataclass
class ArcCsch(Base):
    child: Base


@dataclass
class ArcSech(Base):
    child: Base


@dataclass
class ArcCoth(Base):
    child: Base


@dataclass
class RateOf(Base):
    target: Base


###############################################################################
# Binary fns
###############################################################################


@dataclass
class Pow(Base):
    left: Base
    right: Base

    def __repr__(self) -> str:
        return f"{self.left!r} ** {self.right!r}"


@dataclass
class Implies(Base):
    left: Base
    right: Base


###############################################################################
# n-ary fns
###############################################################################


@dataclass
class Function(Base):
    name: str
    children: list[Base]


@dataclass
class Max(Base):
    children: list[Base]


@dataclass
class Min(Base):
    children: list[Base]


@dataclass
class Piecewise(Base):
    children: list[Base]


@dataclass
class Rem(Base):
    children: list[Base]


@dataclass
class Lambda(Base):
    fn: Base
    args: list[Base]


@dataclass
class And(Base):
    children: list[Base]


@dataclass
class Not(Base):
    children: list[Base]


@dataclass
class Or(Base):
    children: list[Base]


@dataclass
class Xor(Base):
    children: list[Base]


@dataclass
class Eq(Base):
    children: list[Base]


@dataclass
class GreaterEqual(Base):
    children: list[Base]


@dataclass
class GreaterThan(Base):
    children: list[Base]


@dataclass
class LessEqual(Base):
    children: list[Base]


@dataclass
class LessThan(Base):
    """
    a < b
    a < b < c
    """

    children: list[Base]


@dataclass
class NotEqual(Base):
    children: list[Base]


@dataclass
class Add(Base):
    children: list[Base]


@dataclass
class Minus(Base):
    children: list[Base]


@dataclass
class Mul(Base):
    children: list[Base]

    def __repr__(self) -> str:
        return " * ".join(repr(i) for i in self.children)


@dataclass
class Divide(Base):
    children: list[Base]


@dataclass
class IntDivide(Base):
    children: list[Base]


@dataclass
class Delay(Base):
    children: list[Base]


__all__ = [
    "AST_TYPES",
    "Abs",
    "Acos",
    "Acot",
    "Add",
    "And",
    "ArcCosh",
    "ArcCoth",
    "ArcCsc",
    "ArcCsch",
    "ArcSec",
    "ArcSech",
    "ArcSinh",
    "ArcTanh",
    "Asin",
    "Atan",
    "Base",
    "Boolean",
    "Ceiling",
    "Constant",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Delay",
    "Divide",
    "Eq",
    "Exp",
    "Factorial",
    "Float",
    "Floor",
    "Function",
    "GreaterEqual",
    "GreaterThan",
    "Implies",
    "IntDivide",
    "Integer",
    "LOGGER",
    "Lambda",
    "LessEqual",
    "LessThan",
    "Ln",
    "Log",
    "Max",
    "Min",
    "Minus",
    "Mul",
    "Not",
    "NotEqual",
    "Or",
    "Piecewise",
    "Pow",
    "RateOf",
    "Rem",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Sqrt",
    "Symbol",
    "Tan",
    "Tanh",
    "Xor",
    "handle_ast_constant_e",
    "handle_ast_constant_false",
    "handle_ast_constant_pi",
    "handle_ast_constant_true",
    "handle_ast_divide",
    "handle_ast_divide_int",
    "handle_ast_function",
    "handle_ast_function_abs",
    "handle_ast_function_ceiling",
    "handle_ast_function_delay",
    "handle_ast_function_exp",
    "handle_ast_function_factorial",
    "handle_ast_function_floor",
    "handle_ast_function_ln",
    "handle_ast_function_log",
    "handle_ast_function_max",
    "handle_ast_function_min",
    "handle_ast_function_piecewise",
    "handle_ast_function_power",
    "handle_ast_function_rate_of",
    "handle_ast_function_rem",
    "handle_ast_function_root",
    "handle_ast_integer",
    "handle_ast_lambda",
    "handle_ast_logical_and",
    "handle_ast_logical_implies",
    "handle_ast_logical_not",
    "handle_ast_logical_or",
    "handle_ast_logical_xor",
    "handle_ast_minus",
    "handle_ast_name",
    "handle_ast_name_avogadro",
    "handle_ast_name_time",
    "handle_ast_originates_in_package",
    "handle_ast_plus",
    "handle_ast_rational",
    "handle_ast_real",
    "handle_ast_relational_eq",
    "handle_ast_relational_geq",
    "handle_ast_relational_gt",
    "handle_ast_relational_leq",
    "handle_ast_relational_lt",
    "handle_ast_relational_neq",
    "handle_ast_times",
    "handle_ast_trigonometric_arc_cos",
    "handle_ast_trigonometric_arc_cosh",
    "handle_ast_trigonometric_arc_cot",
    "handle_ast_trigonometric_arc_coth",
    "handle_ast_trigonometric_arc_csc",
    "handle_ast_trigonometric_arc_csch",
    "handle_ast_trigonometric_arc_sec",
    "handle_ast_trigonometric_arc_sech",
    "handle_ast_trigonometric_arc_sin",
    "handle_ast_trigonometric_arc_sinh",
    "handle_ast_trigonometric_arc_tan",
    "handle_ast_trigonometric_arc_tanh",
    "handle_ast_trigonometric_cos",
    "handle_ast_trigonometric_cosh",
    "handle_ast_trigonometric_cot",
    "handle_ast_trigonometric_coth",
    "handle_ast_trigonometric_csc",
    "handle_ast_trigonometric_csch",
    "handle_ast_trigonometric_sec",
    "handle_ast_trigonometric_sech",
    "handle_ast_trigonometric_sin",
    "handle_ast_trigonometric_sinh",
    "handle_ast_trigonometric_tan",
    "handle_ast_trigonometric_tanh",
    "parse_sbml_math",
]


AST_TYPES = get_ast_types()


def handle_ast_constant_e(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    return Constant("e")


def handle_ast_constant_false(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    return Boolean(value=False)


def handle_ast_constant_true(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    return Boolean(value=True)


def handle_ast_constant_pi(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    return Constant("pi")


def handle_ast_function(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    name = node.getName()
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("name: %s, args: %s", name, children)
    return Function(name=name, children=children)


def handle_ast_function_abs(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Abs(child=child)


def handle_ast_function_ceiling(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Ceiling(child=child)


def handle_ast_function_delay(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    return Delay(children=children)


def handle_ast_function_exp(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Exp(child=child)


def handle_ast_function_factorial(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Factorial(child=child)


def handle_ast_function_floor(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Floor(child=child)


def handle_ast_function_ln(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Ln(child=child)


def handle_ast_function_log(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    base = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    arg = _handle_ast_node(node=node.getChild(1), func_arguments=func_arguments)
    LOGGER.debug("base: %s, child: %s", base, arg)
    return Log(base=base, child=arg)


def handle_ast_function_max(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Max(children=children)


def handle_ast_function_min(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Min(children=children)


def handle_ast_function_piecewise(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    """
    <piecewise>
    <piece> value condition </piece>
        ....
    <piece> value condition </piece>
    <otherwise> value </otherwise>
    </piecewise>
    """
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Piecewise(children=children)


def handle_ast_function_power(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Pow(left=children[0], right=children[1])


def handle_ast_function_rate_of(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return RateOf(target=child)


def handle_ast_function_root(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    base = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    child = _handle_ast_node(node=node.getChild(1), func_arguments=func_arguments)
    LOGGER.debug("child: %s", child)
    return Sqrt(base=base, child=child)


def handle_ast_function_rem(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return Rem(children=children)


def handle_ast_integer(
    node: ASTNode,
    func_arguments: list[Symbol],  # noqa: ARG001,
) -> Base:
    child = node.getValue()
    LOGGER.debug("child: %s", child)
    return Integer(value=child)


def handle_ast_lambda(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    """
    A function having n variables is specified using a lambda element with n + 1 child elements.
    Each of the first n child elements is a bvar element that represents one of the variables of the function
    The (n + 1)th element is an expression that defines the function itself
    """
    # num_b_vars = node.getNumBvars()
    num_children = node.getNumChildren()
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(num_children)
    ]
    LOGGER.debug("children: %s", children)
    return Lambda(args=children[:-1], fn=children[-1])


def handle_ast_logical_and(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return And(children=children)


def handle_ast_logical_implies(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    return Implies(children[0], children[1])


def handle_ast_logical_not(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Not(children=children)


def handle_ast_logical_or(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Or(children=children)


def handle_ast_logical_xor(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Xor(children=children)


def handle_ast_plus(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Add(children=children)


def handle_ast_minus(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Minus(children=children)


def handle_ast_times(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Mul(children=children)


def handle_ast_divide(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Divide(children=children)


def handle_ast_divide_int(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return IntDivide(children=children)


def handle_ast_name(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    name = Symbol(name_to_py(node.getName()))
    if name not in func_arguments:
        func_arguments.append(name)
    LOGGER.debug("name: %s", name)
    return name


def handle_ast_name_avogadro(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],  # noqa: ARG001,
) -> Base:
    return Float(6.02214179e23)


def handle_ast_name_time(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[Symbol],
) -> Base:
    name = Symbol("time")
    if name not in func_arguments:
        func_arguments.append(name)
    return name


def handle_ast_originates_in_package(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    raise NotImplementedError


def handle_ast_rational(
    node: ASTNode,
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    val = node.getValue()
    LOGGER.debug("val: %s", val)
    return Float(value=val)


def handle_ast_real(
    node: ASTNode,
    func_arguments: list[Symbol],  # noqa: ARG001
) -> Base:
    value = str(node.getValue())
    LOGGER.debug("value: %s", value)
    if value == "inf":
        return Float(math.inf)
    if value == "nan":
        return Float(math.nan)

    # FIXME: seems dumb
    return Float(value=float(value))


def handle_ast_relational_eq(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return Eq(children=children)


def handle_ast_relational_geq(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return GreaterEqual(children=children)


def handle_ast_relational_gt(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return GreaterThan(children=children)


def handle_ast_relational_leq(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return LessEqual(children=children)


def handle_ast_relational_lt(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    LOGGER.debug("children: %s", children)
    return LessThan(children=children)


def handle_ast_relational_neq(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]

    LOGGER.debug("children: %s", children)
    return NotEqual(children=children)


###############################################################################
# Base
###############################################################################


def handle_ast_trigonometric_sin(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Sin(child=child)


def handle_ast_trigonometric_cos(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Cos(child=child)


def handle_ast_trigonometric_tan(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Tan(child=child)


def handle_ast_trigonometric_sec(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Sec(child)


def handle_ast_trigonometric_csc(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Csc(child)


def handle_ast_trigonometric_cot(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Cot(child)


###############################################################################
# Inverse
###############################################################################


def handle_ast_trigonometric_arc_sin(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Asin(child)


def handle_ast_trigonometric_arc_cos(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Acos(child)


def handle_ast_trigonometric_arc_tan(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Atan(child)


def handle_ast_trigonometric_arc_cot(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Acot(child)


def handle_ast_trigonometric_arc_sec(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcSec(child)


def handle_ast_trigonometric_arc_csc(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcCsc(child)


###############################################################################
# Hyperbolic
###############################################################################


def handle_ast_trigonometric_sinh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Sinh(child)


def handle_ast_trigonometric_cosh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Cosh(child)


def handle_ast_trigonometric_tanh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Tanh(child)


def handle_ast_trigonometric_sech(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Sech(child)


def handle_ast_trigonometric_csch(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Csch(child)


def handle_ast_trigonometric_coth(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return Coth(child)


###############################################################################
# Hyperbolic - inverse
###############################################################################


def handle_ast_trigonometric_arc_sinh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcSinh(child)


def handle_ast_trigonometric_arc_cosh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcCosh(child)


def handle_ast_trigonometric_arc_tanh(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcTanh(child)


def handle_ast_trigonometric_arc_csch(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcCsch(child)


def handle_ast_trigonometric_arc_sech(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return ArcSech(child)


def handle_ast_trigonometric_arc_coth(
    node: ASTNode,
    func_arguments: list[Symbol],
) -> Base:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)

    return ArcCoth(child)


def _handle_ast_node(node: ASTNode, func_arguments: list[Symbol]) -> Base:
    commands = {
        "AST_CONSTANT_E": handle_ast_constant_e,
        "AST_CONSTANT_FALSE": handle_ast_constant_false,
        "AST_CONSTANT_PI": handle_ast_constant_pi,
        "AST_CONSTANT_TRUE": handle_ast_constant_true,
        "AST_DIVIDE": handle_ast_divide,
        "AST_FUNCTION": handle_ast_function,
        "AST_FUNCTION_ABS": handle_ast_function_abs,
        "AST_FUNCTION_ARCCOS": handle_ast_trigonometric_arc_cos,
        "AST_FUNCTION_ARCCOSH": handle_ast_trigonometric_arc_cosh,
        "AST_FUNCTION_ARCCOT": handle_ast_trigonometric_arc_cot,
        "AST_FUNCTION_ARCCOTH": handle_ast_trigonometric_arc_coth,
        "AST_FUNCTION_ARCCSC": handle_ast_trigonometric_arc_csc,
        "AST_FUNCTION_ARCCSCH": handle_ast_trigonometric_arc_csch,
        "AST_FUNCTION_ARCSEC": handle_ast_trigonometric_arc_sec,
        "AST_FUNCTION_ARCSECH": handle_ast_trigonometric_arc_sech,
        "AST_FUNCTION_ARCSIN": handle_ast_trigonometric_arc_sin,
        "AST_FUNCTION_ARCSINH": handle_ast_trigonometric_arc_sinh,
        "AST_FUNCTION_ARCTAN": handle_ast_trigonometric_arc_tan,
        "AST_FUNCTION_ARCTANH": handle_ast_trigonometric_arc_tanh,
        "AST_FUNCTION_CEILING": handle_ast_function_ceiling,
        "AST_FUNCTION_COS": handle_ast_trigonometric_cos,
        "AST_FUNCTION_COSH": handle_ast_trigonometric_cosh,
        "AST_FUNCTION_COT": handle_ast_trigonometric_cot,
        "AST_FUNCTION_COTH": handle_ast_trigonometric_coth,
        "AST_FUNCTION_CSC": handle_ast_trigonometric_csc,
        "AST_FUNCTION_CSCH": handle_ast_trigonometric_csch,
        "AST_FUNCTION_DELAY": handle_ast_function_delay,
        "AST_FUNCTION_EXP": handle_ast_function_exp,
        "AST_FUNCTION_FACTORIAL": handle_ast_function_factorial,
        "AST_FUNCTION_FLOOR": handle_ast_function_floor,
        "AST_FUNCTION_LN": handle_ast_function_ln,
        "AST_FUNCTION_LOG": handle_ast_function_log,
        "AST_FUNCTION_MAX": handle_ast_function_max,
        "AST_FUNCTION_MIN": handle_ast_function_min,
        "AST_FUNCTION_PIECEWISE": handle_ast_function_piecewise,
        "AST_FUNCTION_POWER": handle_ast_function_power,
        "AST_FUNCTION_QUOTIENT": handle_ast_divide_int,
        "AST_FUNCTION_RATE_OF": handle_ast_function_rate_of,
        "AST_FUNCTION_ROOT": handle_ast_function_root,
        "AST_FUNCTION_REM": handle_ast_function_rem,
        "AST_FUNCTION_SEC": handle_ast_trigonometric_sec,
        "AST_FUNCTION_SECH": handle_ast_trigonometric_sech,
        "AST_FUNCTION_SIN": handle_ast_trigonometric_sin,
        "AST_FUNCTION_SINH": handle_ast_trigonometric_sinh,
        "AST_FUNCTION_TAN": handle_ast_trigonometric_tan,
        "AST_FUNCTION_TANH": handle_ast_trigonometric_tanh,
        "AST_INTEGER": handle_ast_integer,
        "AST_LAMBDA": handle_ast_lambda,
        "AST_LOGICAL_AND": handle_ast_logical_and,
        "AST_LOGICAL_IMPLIES": handle_ast_logical_implies,
        "AST_LOGICAL_NOT": handle_ast_logical_not,
        "AST_LOGICAL_OR": handle_ast_logical_or,
        "AST_LOGICAL_XOR": handle_ast_logical_xor,
        "AST_MINUS": handle_ast_minus,
        "AST_NAME": handle_ast_name,
        "AST_NAME_AVOGADRO": handle_ast_name_avogadro,
        "AST_NAME_TIME": handle_ast_name_time,
        "AST_ORIGINATES_IN_PACKAGE": handle_ast_originates_in_package,
        "AST_PLUS": handle_ast_plus,
        "AST_POWER": handle_ast_function_power,
        "AST_RATIONAL": handle_ast_rational,
        "AST_REAL": handle_ast_real,
        "AST_REAL_E": handle_ast_real,
        "AST_RELATIONAL_EQ": handle_ast_relational_eq,
        "AST_RELATIONAL_GEQ": handle_ast_relational_geq,
        "AST_RELATIONAL_GT": handle_ast_relational_gt,
        "AST_RELATIONAL_LEQ": handle_ast_relational_leq,
        "AST_RELATIONAL_LT": handle_ast_relational_lt,
        "AST_RELATIONAL_NEQ": handle_ast_relational_neq,
        "AST_TIMES": handle_ast_times,
    }
    return commands[AST_TYPES[node.getType()]](node=node, func_arguments=func_arguments)


def parse_sbml_math(node: ASTNode) -> tuple[Base, list[Symbol]]:
    func_arguments: list[Any] = []
    try:
        body = _handle_ast_node(node=node, func_arguments=func_arguments)
    except TypeError as e:
        msg = f"Cannot parse rule {libsbml.formulaToL3String(node)}"
        raise TypeError(msg) from e

    LOGGER.debug("%s", func_arguments)
    return body, func_arguments
