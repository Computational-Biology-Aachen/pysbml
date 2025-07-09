import math

time: float = 0.0
k1: float = 1.00000000000000
k2: float = 0.750000000000000
P1: float = 1.50000000000000
P2: float = 0.0

# Initial assignments
dP1 = P2 * k2 * math.exp(-time)
dP2 = P1 * k1 * math.exp(-time)
y0 = [P1, P2]
variable_names = ["P1", "P2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    P1, P2 = variables
    dP1: float = P2 * k2 * math.exp(-time)
    dP2: float = P1 * k1 * math.exp(-time)
    dP1dt: float = dP1
    dP2dt: float = dP2
    return dP1dt, dP2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    P1, P2 = variables
    dP1: float = P2 * k2 * math.exp(-time)
    dP2: float = P1 * k1 * math.exp(-time)
    return {
        "dP1": dP1,
        "dP2": dP2,
    }
