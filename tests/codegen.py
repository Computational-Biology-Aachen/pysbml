from pathlib import Path

from pysbml import load_document
from pysbml.codegen import codegen
from pysbml.transform import transform

# for folder in sorted((Path(__file__).parent / "assets").iterdir()):
#     if not (f := folder / "model.py").exists():
#         try:
#             doc = load_document(folder / f"{folder.name}-sbml-l3v2.xml")
#             code = codegen(transform(doc))
#             with f.open("w+") as fp:
#                 fp.write(code)
#         except Exception as e:
#             print(folder.name)
#             print(e)

start = """\
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from collections.abc import Iterable

ASSET_PATH = Path(__file__).parent / "assets"

@dataclass
class SimSettings:
    atol: float
    rtol: float
    ids: list[str] | None


def get_simulation_settings(path: Path, prefix: str) -> SimSettings:
    sim_settings = SimSettings(atol=1e-6, rtol=1e-6, ids=None)

    if (settings_file := path / f"{{prefix}}-settings.txt").exists():
        with settings_file.open() as f:
            for line in f:
                i = line.strip().split(": ")
                if i[0] == "absolute":
                    sim_settings.atol = float(i[1])
                elif i[0] == "relative":
                    sim_settings.rtol = float(i[1])
                elif i[0] == "amount":
                    sim_settings.ids = [j.strip() for j in i[1].split(",")]
    return sim_settings


current = Path(__file__).parent
settings = get_simulation_settings(path=current, prefix="{prefix}")
expected = pd.read_csv(current / "{prefix}-results.csv", index_col=0).astype(float)
expected.columns = [i.strip() for i in expected.columns]
t = expected.index

### Start generated
{model}
### end generated

sol = solve_ivp(
    model,
    y0=y0,
    t_span=(t[0], t[-1]),
    t_eval=t,
    atol=settings.atol / 100,
    rtol=settings.rtol / 100,
    method="LSODA",
)

result = pd.DataFrame(data=sol.y.T, index=sol.t, columns=variable_names)
result = pd.concat(
    (
        result,
        pd.DataFrame(
            data=[derived(cast(float, time), res) for time, res in result.iterrows()],
            index=sol.t,
        ).astype(float),
    ),
    axis=1,
)
columns: list[str] = list(expected.columns.intersection(result.columns))


def test_simulation() -> None:
    np.testing.assert_allclose(
        result.loc[:, columns],
        expected.loc[:, columns],
        rtol=1e-2,
        atol=1e-4,
        err_msg=f"Failed test {prefix}",
    )


if __name__ == "__main__":
    test_simulation()

"""

for folder in sorted((Path(__file__).parent / "assets").iterdir()):
    test = folder.name
    f = folder / f"test_{test}.py"
    try:
        doc = load_document(folder / f"{test}-sbml-l3v2.xml")
        code = codegen(transform(doc))
        with f.open("w+") as fp:
            fp.write(start.format(prefix=test, model=code))
    except Exception:  # noqa: BLE001 ; fine in tests
        with f.open("w+") as fp:
            fp.write(start.format(prefix=test, model="# FIXME: implement"))

# Create a test_reference.py here, which generates the test and runs the integration
