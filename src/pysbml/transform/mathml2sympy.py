from __future__ import annotations

import logging
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import sympy

from pysbml.parse import mathml

__all__ = ["LOGGER", "convert_mathml"]

if TYPE_CHECKING:
    from pysbml.transform import data

LOGGER = logging.getLogger(__name__)


def _handle_node(node: mathml.Base, fns: dict[str, data.Function]) -> Any:
    match node:
        case mathml.Symbol(name) | mathml.Constant(name):
            return sympy.Symbol(name)
        case mathml.Boolean(value):
            return sympy.true if value else sympy.false
        case mathml.Integer(value):
            return sympy.Integer(value)
        case mathml.Float(value):
            return sympy.Float(value)
        # unary
        case mathml.Abs(value):
            return sympy.Abs(_handle_node(value, fns))
        case mathml.Ceiling(value):
            return sympy.ceiling(_handle_node(value, fns))
        case mathml.Exp(value):
            return sympy.exp(_handle_node(value, fns))
        case mathml.Factorial(value):
            return sympy.factorial(_handle_node(value, fns))
        case mathml.Floor(value):
            return sympy.floor(_handle_node(value, fns))
        case mathml.Ln(value):
            return sympy.ln(_handle_node(value, fns))
        case mathml.Log(value):
            return sympy.log(_handle_node(value, fns), 10)
        case mathml.Sqrt(value):
            return sympy.sqrt(_handle_node(value, fns))
        case mathml.Sin(value):
            return sympy.sin(_handle_node(value, fns))
        case mathml.Cos(value):
            return sympy.cos(_handle_node(value, fns))
        case mathml.Tan(value):
            return sympy.tan(_handle_node(value, fns))
        case mathml.Sec(value):
            return sympy.sec(_handle_node(value, fns))
        case mathml.Csc(value):
            return sympy.csc(_handle_node(value, fns))
        case mathml.Cot(value):
            return sympy.cot(_handle_node(value, fns))
        case mathml.Asin(value):
            return sympy.asin(_handle_node(value, fns))
        case mathml.Acos(value):
            return sympy.acos(_handle_node(value, fns))
        case mathml.Atan(value):
            return sympy.atan(_handle_node(value, fns))
        case mathml.Acot(value):
            return sympy.acot(_handle_node(value, fns))
        case mathml.ArcSec(value):
            return sympy.asec(_handle_node(value, fns))
        case mathml.ArcCsc(value):
            return sympy.acsc(_handle_node(value, fns))
        case mathml.Sinh(value):
            return sympy.sinh(_handle_node(value, fns))
        case mathml.Cosh(value):
            return sympy.cosh(_handle_node(value, fns))
        case mathml.Tanh(value):
            return sympy.tanh(_handle_node(value, fns))
        case mathml.Sech(value):
            return sympy.sech(_handle_node(value, fns))
        case mathml.Csch(value):
            return sympy.csch(_handle_node(value, fns))
        case mathml.Coth(value):
            return sympy.coth(_handle_node(value, fns))
        case mathml.ArcSinh(value):
            return sympy.asinh(_handle_node(value, fns))
        case mathml.ArcCosh(value):
            return sympy.acosh(_handle_node(value, fns))
        case mathml.ArcTanh(value):
            return sympy.atanh(_handle_node(value, fns))
        case mathml.ArcCsch(value):
            return sympy.acsch(_handle_node(value, fns))
        case mathml.ArcSech(value):
            return sympy.asech(_handle_node(value, fns))
        case mathml.ArcCoth(value):
            return sympy.acoth(_handle_node(value, fns))
        case mathml.Lambda(fn, args):
            return sympy.Lambda(
                tuple(_handle_node(i, fns) for i in args), _handle_node(fn, fns)
            )
        # binary
        case mathml.Pow(left, right):
            return sympy.Pow(_handle_node(left, fns), _handle_node(right, fns))
        case mathml.Implies(left, right):
            raise NotImplementedError
        # n-ary
        case mathml.Function(name, children):
            fn = fns[name].body
            return fn(*(_handle_node(i, fns) for i in children))  # type: ignore
        case mathml.Max(children):
            return sympy.Max(*(_handle_node(i, fns) for i in children))
        case mathml.Min(children):
            return sympy.Min(*(_handle_node(i, fns) for i in children))
        case mathml.Piecewise(children):
            handled = [_handle_node(i, fns) for i in children]
            pairs = [
                (handled[2 * i], handled[2 * i + 1]) for i in range(len(children) // 2)
            ]
            return sympy.Piecewise(*pairs, (handled[-1], True))
        case mathml.Rem(children):
            return sympy.rem(*(_handle_node(i, fns) for i in children))
        case mathml.And(children):
            return sympy.And(*(_handle_node(i, fns) for i in children))
        case mathml.Not(children):
            return sympy.Not(*(_handle_node(i, fns) for i in children))
        case mathml.Or(children):
            return sympy.Or(*(_handle_node(i, fns) for i in children))
        case mathml.Xor(children):
            return sympy.Xor(*(_handle_node(i, fns) for i in children))
        case mathml.Eq(children):
            return sympy.Eq(*(_handle_node(i, fns) for i in children))
        case mathml.GreaterEqual(children):
            return sympy.Ge(*(_handle_node(i, fns) for i in children))
        case mathml.GreaterThan(children):
            return sympy.Gt(*(_handle_node(i, fns) for i in children))
        case mathml.LessEqual(children):
            return sympy.Le(*(_handle_node(i, fns) for i in children))
        case mathml.LessThan(children):
            return sympy.Lt(*(_handle_node(i, fns) for i in children))
        case mathml.NotEqual(children):
            return sympy.Ne(*(_handle_node(i, fns) for i in children))
        case mathml.Add(children):
            return sympy.Add(*(_handle_node(i, fns) for i in children))
        case mathml.Mul(children):
            return sympy.Mul(*(_handle_node(i, fns) for i in children))
        # These need special handling as sympy doesn't have a nice
        # constructor
        case mathml.Minus(children):
            children = [_handle_node(i, fns) for i in children]
            if len(children) == 1:
                return -children[0]
            return reduce(operator.sub, children)
        case mathml.Divide(children):
            return reduce(operator.truediv, (_handle_node(i, fns) for i in children))
        case mathml.IntDivide(children):
            return reduce(operator.floordiv, (_handle_node(i, fns) for i in children))
        case _:
            raise NotImplementedError(type(node))


def convert_mathml(node: mathml.Base, fns: dict[str, data.Function]) -> sympy.Expr:
    return _handle_node(node=node, fns=fns)
