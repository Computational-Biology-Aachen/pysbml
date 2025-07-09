time: float = 0.0
c: float = 1.00000000000000
s: float = 2.00000000000000

# Initial assignments
s_conc = s / c
dc = 0.5 * c
y0 = [c]
variable_names = ["c"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (c,) = variables
    dc: float = 0.5 * c
    dcdt: float = dc
    return (dcdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (c,) = variables
    s_conc: float = s / c
    dc: float = 0.5 * c
    return {
        "s_conc": s_conc,
        "dc": dc,
    }
