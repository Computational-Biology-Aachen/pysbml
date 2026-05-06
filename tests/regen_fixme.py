"""Regenerate test files that currently have '# FIXME: implement'."""

from __future__ import annotations

from pathlib import Path

from pysbml import load_document
from pysbml.codegen import codegen
from pysbml.transform import transform

# Templates use __PREFIX__ and __MODEL__ as unique sentinels to avoid
# conflicts with Python's str.format() and curly braces in generated code.

SIMPLE_TEMPLATE = """\
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


current = Path(__file__).parent
settings = get_simulation_settings(path=current, prefix="__PREFIX__")
expected = pd.read_csv(current / "__PREFIX__-results.csv", index_col=0).astype(float)
expected.columns = [i.strip() for i in expected.columns]
t = expected.index

### Start generated
__MODEL__
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
        err_msg=f"Failed test __PREFIX__",
    )


if __name__ == "__main__":
    test_simulation()
"""

EVENT_TEMPLATE = """\
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


current = Path(__file__).parent
settings = get_simulation_settings(path=current, prefix="__PREFIX__")
expected = pd.read_csv(current / "__PREFIX__-results.csv", index_col=0).astype(float)
expected.columns = [i.strip() for i in expected.columns]
t = expected.index

### Start generated
__MODEL__
### end generated



def _simulate_events(model, y0, t_eval, events, atol, rtol):
    t_arr = np.asarray(t_eval, dtype=float)
    t_end = float(t_arr[-1])
    triggers = [e[0] for e in events]
    assign_fns = [e[1] for e in events]
    delay_fns = [e[4] for e in events]

    y_cur = np.array(y0, dtype=float)
    t_cur = float(t_arr[0])
    all_t: list = []
    all_y: list = []
    pending: list = []  # (t_fire, vals_or_fn)
    t_eval_done = 0

    # Apply events with initialValue=False triggered at t=0 (priority-ordered)
    y0_state = y_cur.copy()
    t0_ev = [i for i, ev in enumerate(events) if not ev[2] and triggers[i](t_cur, y_cur) > 0]
    while t0_ev:
        ev_pri = [(i, float(events[i][5](t_cur, y_cur)) if events[i][5] is not None else 0.0) for i in t0_ev]
        ev_pri.sort(key=lambda x: -x[1])
        ev_i = ev_pri[0][0]
        t0_ev.remove(ev_i)
        persistent = events[ev_i][3]
        if not persistent and triggers[ev_i](t_cur, y_cur) <= 0:
            continue
        dfn = delay_fns[ev_i]
        d = float(dfn(t_cur, y_cur)) if dfn is not None else 0.0
        uvftt = events[ev_i][6]
        eval_state = y0_state if uvftt else y_cur
        if d > 0:
            if uvftt:
                vals = assign_fns[ev_i](t_cur, y0_state)
                pending.append((t_cur + d, vals))
            else:
                pending.append((t_cur + d, assign_fns[ev_i]))
        else:
            vals = assign_fns[ev_i](t_cur, eval_state)
            for var, val in vals.items():
                y_cur[variable_names.index(var)] = val

    for _ in range(100000):
        if t_eval_done >= len(t_arr) and not pending:
            break

        t_next_p = min((p[0] for p in pending), default=t_end)
        t_seg = min(t_end, t_next_p)

        seg = t_arr[t_eval_done:]
        seg_in_range = seg[seg <= t_seg + 1e-12]

        avail_idx = [i for i, trig in enumerate(triggers)
                     if trig(t_cur, y_cur) <= 0]
        avail_triggers = [triggers[i] for i in avail_idx]

        if len(seg_in_range) > 0:
            try:
                sol = solve_ivp(
                    model, y0=y_cur,
                    t_span=(t_cur, t_seg),
                    t_eval=seg_in_range,
                    events=avail_triggers if avail_triggers else None,
                    atol=atol, rtol=rtol, method="LSODA",
                )
            except ValueError:
                # brentq failure: trigger is at boundary at t_cur (LSODA dense
                # output backward-extrapolates past the discontinuity). Advance
                # past the boundary with a tiny ODE step and retry.
                _eps = max(1e-12, 1e-9 * abs(t_seg - t_cur))
                _sol_eps = solve_ivp(
                    model, y0=y_cur, t_span=(t_cur, t_cur + _eps),
                    atol=atol, rtol=rtol, method="LSODA",
                )
                if _sol_eps.status == 0 and len(_sol_eps.t) > 0:
                    y_cur = _sol_eps.y[:, -1].copy()
                    t_cur = float(_sol_eps.t[-1])
                else:
                    t_cur += _eps
                continue
            sol_t = np.asarray(sol.t, dtype=float)
            sol_y = np.asarray(sol.y, dtype=float)
            all_t.extend(sol_t.tolist())
            all_y.extend(sol_y.T.tolist())
            t_eval_done += len(sol_t)
            if len(sol_t) > 0:
                t_cur = float(sol_t[-1])
                y_cur = sol_y[:, -1].copy()

            if sol.status == 1:
                # Find earliest event time from scipy detections
                first_t_ev, first_ye = t_end, None
                for j, (te, ye) in enumerate(zip(sol.t_events, sol.y_events)):
                    if te.size > 0 and te[0] < first_t_ev - 1e-12:
                        first_t_ev, first_ye = te[0], ye[0].copy()
                if first_ye is not None:
                    # Collect ALL available events triggered at first_t_ev
                    # (scipy may miss simultaneous events due to terminal stopping)
                    remaining_simult = [
                        ev_i for ev_i in avail_idx
                        if triggers[ev_i](first_t_ev, first_ye) > 0
                    ]
                    y_new = first_ye.copy()
                    # Fire in priority order, re-evaluating priorities after each firing
                    while remaining_simult:
                        ev_pri = [
                            (ev_i, float(events[ev_i][5](first_t_ev, y_new)) if events[ev_i][5] is not None else 0.0)
                            for ev_i in remaining_simult
                        ]
                        ev_pri.sort(key=lambda x: -x[1])
                        ev_i, _ = ev_pri[0]
                        remaining_simult.remove(ev_i)
                        persistent = events[ev_i][3]
                        # persistent=False: re-check trigger with updated state
                        if not persistent and triggers[ev_i](first_t_ev, y_new) <= 0:
                            continue
                        dfn = delay_fns[ev_i]
                        d = float(dfn(first_t_ev, first_ye)) if dfn is not None else 0.0
                        uvftt = events[ev_i][6]
                        eval_state = first_ye if uvftt else y_new
                        if d > 0:
                            if uvftt:
                                vals = assign_fns[ev_i](first_t_ev, first_ye)
                                pending.append((first_t_ev + d, vals))
                            else:
                                pending.append((first_t_ev + d, assign_fns[ev_i]))
                        else:
                            vals = assign_fns[ev_i](first_t_ev, eval_state)
                            for var, val in vals.items():
                                y_new[variable_names.index(var)] = val
                    t_cur = first_t_ev
                    y_cur = y_new

            elif t_cur < t_seg - 1e-10:
                sol_fill = solve_ivp(
                    model, y0=y_cur,
                    t_span=(t_cur, t_seg),
                    t_eval=[t_seg],
                    atol=atol, rtol=rtol, method="LSODA",
                )
                y_cur = sol_fill.y[:, -1].copy()
                t_cur = t_seg

        else:
            sol_fill = solve_ivp(
                model, y0=y_cur,
                t_span=(t_cur, t_seg),
                t_eval=[t_seg],
                atol=atol, rtol=rtol, method="LSODA",
            )
            y_cur = sol_fill.y[:, -1].copy()
            t_cur = t_seg

        new_pending = []
        for p_t, p_v in pending:
            if p_t <= t_cur + 1e-10:
                apply_vals = p_v(t_cur, y_cur) if callable(p_v) else p_v
                for var, val in apply_vals.items():
                    y_cur = y_cur.copy()
                    y_cur[variable_names.index(var)] = val
            else:
                new_pending.append((p_t, p_v))
        pending = new_pending

        if t_cur >= t_end:
            break

    return np.array(all_t), np.array(all_y).T

sim_t, sim_y = _simulate_events(
    model, y0=y0, t_eval=t,
    events=events,
    atol=settings.atol / 100,
    rtol=settings.rtol / 100,
)

result = pd.DataFrame(data=sim_y.T, index=sim_t, columns=variable_names)
result = pd.concat(
    (
        result,
        pd.DataFrame(
            data=[derived(cast(float, time), res) for time, res in result.iterrows()],
            index=sim_t,
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
        err_msg=f"Failed test __PREFIX__",
    )


if __name__ == "__main__":
    test_simulation()
"""


SKIP_TEMPLATE = """\
import pytest


def test_simulation() -> None:
    pytest.skip("__REASON__")
"""


def is_fixme(test_file: Path) -> bool:
    if not test_file.exists():
        return True
    content = test_file.read_text()
    return "# FIXME: implement" in content or "return {{" in content


def has_events(code: str) -> bool:
    if "events: list = [" not in code:
        return False
    after = code.split("events: list = [", 1)[1]
    return not after.startswith("]")


def is_ode_empty(code: str) -> bool:
    """True when model has no ODE variables (e.g. FBC flux-balance models)."""
    return "y0 = []" in code


def render(template: str, prefix: str, model: str) -> str:
    return template.replace("__PREFIX__", prefix).replace("__MODEL__", model)


ok = 0
failed = []

for folder in sorted((Path(__file__).parent / "assets").iterdir()):
    test = folder.name
    f = folder / f"test_{test}.py"

    if not is_fixme(f):
        continue

    sbml_file = folder / f"{test}-sbml-l3v2.xml"
    if not sbml_file.exists():
        continue

    try:
        doc = load_document(sbml_file)
        code = codegen(transform(doc))

        if is_ode_empty(code):
            f.write_text(
                SKIP_TEMPLATE.replace(
                    "__REASON__", "FBC/FBA not supported (requires LP solver)"
                )
            )
        else:
            tmpl = EVENT_TEMPLATE if has_events(code) else SIMPLE_TEMPLATE
            f.write_text(render(tmpl, test, code))
        ok += 1
        print(f"OK  {test}")
    except Exception as e:  # noqa: BLE001
        failed.append((test, str(e)))
        print(f"ERR {test}: {e}")

print(f"\n{ok} regenerated, {len(failed)} still failed")
if failed:
    print("Still failing:")
    for t, e in failed[:20]:
        print(f"  {t}: {e}")
