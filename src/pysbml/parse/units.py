from __future__ import annotations

import libsbml

__all__ = ["get_ast_types", "get_operator_mappings", "get_unit_conversion"]

_unit_conversion = {
    "UNIT_KIND_AMPERE": "AMPERE",
    "UNIT_KIND_AVOGADRO": "AVOGADRO",
    "UNIT_KIND_BECQUEREL": "BECQUEREL",
    "UNIT_KIND_CANDELA": "CANDELA",
    "UNIT_KIND_CELSIUS": "CELSIUS",
    "UNIT_KIND_COULOMB": "COULOMB",
    "UNIT_KIND_DIMENSIONLESS": "DIMENSIONLESS",
    "UNIT_KIND_FARAD": "FARAD",
    "UNIT_KIND_GRAM": "GRAM",
    "UNIT_KIND_GRAY": "GRAY",
    "UNIT_KIND_HENRY": "HENRY",
    "UNIT_KIND_HERTZ": "HERTZ",
    "UNIT_KIND_ITEM": "ITEM",
    "UNIT_KIND_JOULE": "JOULE",
    "UNIT_KIND_KATAL": "KATAL",
    "UNIT_KIND_KELVIN": "KELVIN",
    "UNIT_KIND_KILOGRAM": "KILOGRAM",
    "UNIT_KIND_LITER": "LITER",
    "UNIT_KIND_LITRE": "LITER",
    "UNIT_KIND_LUMEN": "LUMEN",
    "UNIT_KIND_LUX": "LUX",
    "UNIT_KIND_METER": "METER",
    "UNIT_KIND_METRE": "METRE",
    "UNIT_KIND_MOLE": "MOLE",
    "UNIT_KIND_NEWTON": "NEWTON",
    "UNIT_KIND_OHM": "OHM",
    "UNIT_KIND_PASCAL": "PASCAL",
    "UNIT_KIND_RADIAN": "RADIAN",
    "UNIT_KIND_SECOND": "SECOND",
    "UNIT_KIND_SIEMENS": "SIEMENS",
    "UNIT_KIND_SIEVERT": "SIEVERT",
    "UNIT_KIND_STERADIAN": "STERADIAN",
    "UNIT_KIND_TESLA": "TESLA",
    "UNIT_KIND_VOLT": "VOLT",
    "UNIT_KIND_WATT": "WATT",
    "UNIT_KIND_WEBER": "WEBER",
    "UNIT_KIND_INVALID": "INVALID",
}


def get_unit_conversion() -> dict[str, str]:
    return {getattr(libsbml, k): v for k, v in _unit_conversion.items()}


def get_ast_types() -> dict[str, str]:
    return dict(
        sorted(
            {
                getattr(libsbml, i): i for i in dir(libsbml) if i.startswith("AST_")
            }.items()
        )
    )


def get_operator_mappings() -> dict[str, str]:
    return {
        "AST_TIMES": "*",
        "AST_PLUS": "+",
        "AST_MINUS": "-",
        "AST_DIVIDE": "/",
        "AST_POWER": "**",
    }
