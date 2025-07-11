# pySBML

[![pypi](https://img.shields.io/pypi/v/pysbml.svg)](https://pypi.python.org/pypi/pysbml)
[![docs][docs-badge]][docs]
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fgist.githubusercontent.com%2Fmarvinvanaalst%2Fb518d017c83b8938be27036ee04df0e3%2Fraw%2F77de2288523103b449f89df15b627431bbfdf403%2Fcoverage.json&query=%24.message&label=Coverage&color=%24.color&suffix=%20%25)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![PyPI Downloads](https://static.pepy.tech/badge/pysbml)](https://pepy.tech/projects/pysbml)

[docs-badge]: https://img.shields.io/badge/docs-main-green.svg?style=flat-square
[docs]: https://computational-biology-aachen.github.io/pysbml/


`pySBML` takes SBML models and makes them simpler :heart:

## Installation

Installation is as easy as `pip install pysbml`.


## Transform this

**Compartments**

| name | size | is_constant |
| ---- | ---- | ----------- |
| C    | 1.0  | True        |

**Variables**

| name | amount | conc    | constant | substance_units | compartment | only_substance_units | boundary_condition |
| ---- | ------ | ------- | -------- | --------------- | ----------- | -------------------- | ------------------ |
| S1   | None   | 0.00015 | False    | substance       | C           | False                | False              |
| S2   | None   | 0.0     | False    | substance       | C           | False                | False              |

**Parameters**

| name | value | is_constant | unit |
| ---- | ----- | ----------- | ---- |
| k1   | 1.0   | True        |      |

**Reactions**

| name      | body        | args        | stoichiometry           | local pars |
| --------- | ----------- | ----------- | ----------------------- | ---------- |
| reaction1 | C * k1 * S1 | [C, k1, S1] | {'S1': -1.0, 'S2': 1.0} | {}         |

## Into this


**Parameters**

| name | value | unit |
| ---- | ----- | ---- |
| k1   | 1.0   | None |
| C    | 1.0   | None |

**Variables**

| name | value   | unit |
| ---- | ------- | ---- |
| S1   | 0.00015 | None |
| S2   | 0.0     | None |

**Reactions**

| name      | fn    | stoichiometry           |
| --------- | ----- | ----------------------- |
| reaction1 | S1*k1 | {'S1': -1.0, 'S2': 1.0} |



## Development setup

We recommend using `uv`

### uv

- Install `uv` as described in [the docs](https://docs.astral.sh/uv/getting-started/installation/).
- Run `uv sync --all-extras --all-groups` to install dependencies locally
