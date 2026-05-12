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

test = "00001"


@dataclass
class SimSettings:
    atol: float
    rtol: float
    ids: list[str] | None


def get_simulation_settings(path: Path, prefix: str) -> SimSettings:
    sim_settings = SimSettings(atol=1e-6, rtol=1e-6, ids=None)

    if (settings_file := path / f"{prefix}-settings.txt").exists():
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


current = Path(__file__)
settings = get_simulation_settings(path=current, prefix=test)
expected = pd.read_csv(current / f"{test}-results.csv", index_col=0).astype(float)
expected.columns = [i.strip() for i in expected.columns]
t = expected.index

### Start generated
time: float = 0.0
k1: float = 1.00000000000000
compartment: float = 1.00000000000000
S1: float = 0.000150000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = S1_conc * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(
    time: float,  # noqa: ARG001 ; API stability
    variables: Iterable[float],
) -> Iterable[float]:
    S1, _S2 = variables
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * k1
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt


def derived(
    time: float,  # noqa: ARG001 ; API stability
    variables: Iterable[float],
) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }


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
        err_msg=f"Failed test {test}",
    )


if __name__ == "__main__":
    test_simulation()
