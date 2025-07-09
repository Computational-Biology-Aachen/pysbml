time: float = 0.0
c: float = 1.00000000000000

# Initial assignments
dc = 0.5 * c
s_amount = 2.0 * c
s = s_amount / c
y0 = [c]
variable_names = ["c"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (c,) = variables
    dc: float = 0.5 * c
    dcdt: float = dc
    return (dcdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (c,) = variables
    s: float = s_amount / c
    dc: float = 0.5 * c
    return {
        "s": s,
        "dc": dc,
    }
