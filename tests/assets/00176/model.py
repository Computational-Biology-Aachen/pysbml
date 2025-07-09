time: float = 0.0
k1: float = 1.00000000000000
S1: float = 0.150000000000000
S2: float = 0.0

# Initial assignments
dS1 = -S1 * k1
dS2 = S1 * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    dS1: float = -S1 * k1
    dS2: float = S1 * k1
    dS1dt: float = dS1
    dS2dt: float = dS2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    dS1: float = -S1 * k1
    dS2: float = S1 * k1
    return {
        "dS1": dS1,
        "dS2": dS2,
    }
