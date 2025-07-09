time: float = 0.0
c: float = 0.500000000000000

# Initial assignments
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
    dc: float = 0.5 * c
    return {
        "dc": dc,
    }
