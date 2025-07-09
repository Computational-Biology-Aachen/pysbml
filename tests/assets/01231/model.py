time: float = 0.0
k1: float = 1.00000000000000
c: float = 1.00000000000000
S1: float = 0.0
S2: float = 0.0

# Initial assignments
J0 = k1
J1 = J0 + 1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    J0: float = k1
    J1: float = J0 + 1
    dS1dt: float = J0
    dS2dt: float = J1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    J0: float = k1
    J1: float = J0 + 1
    return {
        "J0": J0,
        "J1": J1,
    }
