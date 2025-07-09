time: float = 0.0
C1: float = 0.0150000000000000
C2: float = 0.0100000000000000

# Initial assignments
dC1 = 0.4 * C1 + 0.25 * C2
dC2 = 0.15 * C1 + 0.2 * C2
y0 = [C1, C2]
variable_names = ["C1", "C2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C1, C2 = variables
    dC1: float = 0.4 * C1 + 0.25 * C2
    dC2: float = 0.15 * C1 + 0.2 * C2
    dC1dt: float = dC1
    dC2dt: float = dC2
    return dC1dt, dC2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    C1, C2 = variables
    dC1: float = 0.4 * C1 + 0.25 * C2
    dC2: float = 0.15 * C1 + 0.2 * C2
    return {
        "dC1": dC1,
        "dC2": dC2,
    }
