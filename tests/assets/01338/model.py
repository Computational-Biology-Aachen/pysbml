time: float = 0.0
C: float = 1.00000000000000
k1: float = 5.00000000000000
S1: float = 0.0

# Initial assignments
dk1 = -1
J0 = k1
y0 = [S1, k1]
variable_names = ["S1", "k1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, k1 = variables
    dk1: float = -1
    J0: float = k1
    dk1dt: float = dk1
    dS1dt: float = J0
    return dS1dt, dk1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, k1 = variables
    dk1: float = -1
    J0: float = k1
    return {
        "dk1": dk1,
        "J0": J0,
    }
