from __future__ import annotations

import itertools as it
import logging
import operator as op
from functools import reduce
from typing import TYPE_CHECKING, Any

import sympy

from pysbml.parse import mathml

__all__ = ["LOGGER", "convert_mathml"]

if TYPE_CHECKING:
    from pysbml.transform import data

LOGGER = logging.getLogger(__name__)


def _handle_node(
    node: mathml.Base,
    fns: dict[str, data.Expr],
    *,
    as_bool: bool = False,
    as_number: bool = False,
) -> Any:
    match node:
        case mathml.Symbol(name):
            return sympy.Symbol(name)
        case mathml.Constant(name):
            if name == "e":
                return sympy.E
            if name == "pi":
                return sympy.pi
            msg = f"Unknown constant {name}"
            raise NotImplementedError(msg)
        case mathml.Boolean(value):
            if as_number:
                return sympy.Float(float(value))
            return sympy.true if value else sympy.false
        case mathml.Integer(value):
            if as_bool:
                return sympy.true if value else sympy.false
            return sympy.Integer(value)
        case mathml.Float(value):
            if as_bool:
                return sympy.true if value else sympy.false
            return sympy.Float(value)
        # unary
        case mathml.Abs(value):
            return sympy.Abs(_handle_node(value, fns, as_number=True))
        case mathml.Ceiling(value):
            return sympy.ceiling(_handle_node(value, fns, as_number=True))
        case mathml.Exp(value):
            return sympy.exp(_handle_node(value, fns, as_number=True))
        case mathml.Factorial(value):
            return sympy.factorial(_handle_node(value, fns, as_number=True))
        case mathml.Floor(value):
            return sympy.floor(_handle_node(value, fns, as_number=True))
        case mathml.Ln(value):
            return sympy.ln(_handle_node(value, fns, as_number=True))
        case mathml.Log(base, value):
            return sympy.log(
                _handle_node(value, fns, as_number=True),
                _handle_node(base, fns, as_number=True),
            )
        case mathml.Sqrt(base, value):
            return sympy.root(
                _handle_node(value, fns, as_number=True),
                _handle_node(base, fns, as_number=True),
            )
        case mathml.Sin(value):
            return sympy.sin(_handle_node(value, fns, as_number=True))
        case mathml.Cos(value):
            return sympy.cos(_handle_node(value, fns, as_number=True))
        case mathml.Tan(value):
            return sympy.tan(_handle_node(value, fns, as_number=True))
        case mathml.Sec(value):
            return sympy.sec(_handle_node(value, fns, as_number=True))
        case mathml.Csc(value):
            return sympy.csc(_handle_node(value, fns, as_number=True))
        case mathml.Cot(value):
            return sympy.cot(_handle_node(value, fns, as_number=True))
        case mathml.Asin(value):
            return sympy.asin(_handle_node(value, fns, as_number=True))
        case mathml.Acos(value):
            return sympy.acos(_handle_node(value, fns, as_number=True))
        case mathml.Atan(value):
            return sympy.atan(_handle_node(value, fns, as_number=True))
        case mathml.Acot(value):
            return sympy.acot(_handle_node(value, fns, as_number=True))
        case mathml.ArcSec(value):
            return sympy.asec(_handle_node(value, fns, as_number=True))
        case mathml.ArcCsc(value):
            return sympy.acsc(_handle_node(value, fns, as_number=True))
        case mathml.Sinh(value):
            return sympy.sinh(_handle_node(value, fns, as_number=True))
        case mathml.Cosh(value):
            return sympy.cosh(_handle_node(value, fns, as_number=True))
        case mathml.Tanh(value):
            return sympy.tanh(_handle_node(value, fns, as_number=True))
        case mathml.Sech(value):
            return sympy.sech(_handle_node(value, fns, as_number=True))
        case mathml.Csch(value):
            return sympy.csch(_handle_node(value, fns, as_number=True))
        case mathml.Coth(value):
            return sympy.coth(_handle_node(value, fns, as_number=True))
        case mathml.ArcSinh(value):
            return sympy.asinh(_handle_node(value, fns, as_number=True))
        case mathml.ArcCosh(value):
            return sympy.acosh(_handle_node(value, fns, as_number=True))
        case mathml.ArcTanh(value):
            return sympy.atanh(_handle_node(value, fns, as_number=True))
        case mathml.ArcCsch(value):
            return sympy.acsch(_handle_node(value, fns, as_number=True))
        case mathml.ArcSech(value):
            return sympy.asech(_handle_node(value, fns, as_number=True))
        case mathml.ArcCoth(value):
            return sympy.acoth(_handle_node(value, fns, as_number=True))
        case mathml.Lambda(fn, args):
            return sympy.Lambda(
                tuple(_handle_node(i, fns) for i in args), _handle_node(fn, fns)
            )
        # binary
        case mathml.Pow(left, right):
            return sympy.Pow(
                _handle_node(left, fns, as_number=True),
                _handle_node(right, fns, as_number=True),
            )
        case mathml.Implies(left, right):
            raise NotImplementedError
        # n-ary
        case mathml.Function(name, children):
            fn = fns[name]
            return fn(*(_handle_node(i, fns) for i in children))  # type: ignore
        case mathml.Max(children):
            return sympy.Max(*(_handle_node(i, fns, as_number=True) for i in children))
        case mathml.Min(children):
            return sympy.Min(*(_handle_node(i, fns, as_number=True) for i in children))
        case mathml.Rem(children):
            return reduce(
                op.mod,
                (_handle_node(i, fns, as_number=True) for i in children),
            )
        case mathml.And(children):
            return sympy.And(*(_handle_node(i, fns, as_bool=True) for i in children))
        case mathml.Not(children):
            return sympy.Not(*(_handle_node(i, fns, as_bool=True) for i in children))
        case mathml.Or(children):
            return sympy.Or(*(_handle_node(i, fns, as_bool=True) for i in children))
        case mathml.Xor(children):
            return sympy.Xor(*(_handle_node(i, fns, as_bool=True) for i in children))
        case mathml.Eq(children):
            return reduce(
                op.and_,
                (
                    sympy.Eq(a, b)
                    for a, b in it.pairwise(_handle_node(i, fns) for i in children)
                ),
            )
        case mathml.NotEqual(children):
            return reduce(
                op.and_,
                (
                    sympy.Unequality(a, b)
                    for a, b in it.pairwise(_handle_node(i, fns) for i in children)
                ),
            )
        case mathml.GreaterEqual(children):
            return reduce(
                op.and_,
                (
                    a >= b
                    for a, b in it.pairwise(_handle_node(i, fns) for i in children)
                ),
            )
        case mathml.GreaterThan(children):
            return reduce(
                op.and_,
                (a > b for a, b in it.pairwise(_handle_node(i, fns) for i in children)),
            )
        case mathml.LessEqual(children):
            return reduce(
                op.and_,
                (
                    a <= b
                    for a, b in it.pairwise(_handle_node(i, fns) for i in children)
                ),
            )
        case mathml.LessThan(children):
            return reduce(
                op.and_,
                (a < b for a, b in it.pairwise(_handle_node(i, fns) for i in children)),
            )
        case mathml.Piecewise(children):
            # Only treat the second arg is necessarily bool
            handled = [
                _handle_node(i, fns)
                if x % 2 == 0
                else _handle_node(i, fns, as_bool=True)
                for x, i in enumerate(children)
            ]
            pairs = [
                (handled[2 * i], handled[2 * i + 1]) for i in range(len(children) // 2)
            ]
            return sympy.Piecewise(*pairs, (handled[-1], True))
        case mathml.Add(children):
            return sympy.Add(*(_handle_node(i, fns, as_number=True) for i in children))
        case mathml.Mul(children):
            return sympy.Mul(*(_handle_node(i, fns, as_number=True) for i in children))
        # These need special handling as sympy doesn't have a nice
        # constructor
        case mathml.Minus(children):
            children = [_handle_node(i, fns, as_number=True) for i in children]
            if len(children) == 1:
                return -children[0]
            return reduce(op.sub, children)
        case mathml.Divide(children):
            return reduce(
                op.truediv, (_handle_node(i, fns, as_number=True) for i in children)
            )
        case mathml.IntDivide(children):
            return reduce(
                op.floordiv, (_handle_node(i, fns, as_number=True) for i in children)
            )
        case _:
            raise NotImplementedError(type(node))


def convert_mathml(node: mathml.Base, fns: dict[str, data.Expr]) -> sympy.Expr:
    return _handle_node(node=node, fns=fns)
