from __future__ import annotations

from sympy import S
from sympy.physics.units import (
    ampere,
    avogadro_number,
    becquerel,
    candela,
    coulomb,
    farad,
    gram,
    gray,
    henry,
    hertz,
    joule,
    katal,
    kelvin,
    kilogram,
    liter,
    lux,
    meter,
    mol,
    newton,
    ohm,
    pascal,
    radian,
    second,
    siemens,
    steradian,
    tesla,
    volt,
    watt,
    weber,
)
from sympy.physics.units.definitions.dimension_definitions import temperature
from sympy.physics.units.prefixes import (
    atto,
    centi,
    deca,
    deci,
    exa,
    femto,
    giga,
    hecto,
    kilo,
    mega,
    micro,
    milli,
    nano,
    peta,
    pico,
    tera,
    yocto,
    yotta,
    zepto,
    zetta,
)
from sympy.physics.units.quantities import Quantity

__all__ = ["CONVERSION", "PREFIXES"]

celsius = Quantity("celsius", abbrev="°C")
celsius.set_global_dimension(temperature)
# we can't do the proper offset here, sympy only allows proportional
# changes
celsius.set_global_relative_scale_factor(S.One, kelvin)

CONVERSION: dict[str, Quantity] = {
    "AMPERE": ampere,
    "AVOGADRO": avogadro_number,
    "BECQUEREL": becquerel,
    "CANDELA": candela,
    "CELSIUS": celsius,
    "COULOMB": coulomb,
    # "DIMENSIONLESS": ,
    "FARAD": farad,
    "GRAM": gram,
    "GRAY": gray,
    "HENRY": henry,
    "HERTZ": hertz,
    # "ITEM": ,
    "JOULE": joule,
    "KATAL": katal,
    "KELVIN": kelvin,
    "KILOGRAM": kilogram,
    "LITER": liter,
    "LITRE": liter,
    # "LUMEN": lumen,
    "LUX": lux,
    "METER": meter,
    "METRE": meter,
    "MOLE": mol,
    "NEWTON": newton,
    "OHM": ohm,
    "PASCAL": pascal,
    "RADIAN": radian,
    "SECOND": second,
    "SIEMENS": siemens,
    # "SIEVERT": sievert,
    "STERADIAN": steradian,
    "TESLA": tesla,
    "VOLT": volt,
    "WATT": watt,
    "WEBER": weber,
    # "INVALID": ,
}

PREFIXES = {
    24: yotta,
    21: zetta,
    18: exa,
    15: peta,
    12: tera,
    9: giga,
    6: mega,
    3: kilo,
    2: hecto,
    1: deca,
    -1: deci,
    -2: centi,
    -3: milli,
    -6: micro,
    -9: nano,
    -12: pico,
    -15: femto,
    -18: atto,
    -21: zepto,
    -24: yocto,
}
