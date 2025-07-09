time: float = 0.0
k1: float = 0.175000000000000
c: float = 22.5000000000000

# Initial assignments
dc = -c / k1
y0 = [c]
variable_names = ["c"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (c,) = variables
    dc: float = -c / k1
    dcdt: float = dc
    return (dcdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (c,) = variables
    dc: float = -c / k1
    return {
        "dc": dc,
    }
