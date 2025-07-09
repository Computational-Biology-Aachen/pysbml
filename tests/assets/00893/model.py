time: float = 0.0
k1: float = 0.500000000000000
P1: float = 0.00150000000000000
P2: float = 0.0
P3: float = 0.00100000000000000

# Initial assignments
dP3 = k1 * time
dP1 = -P1 * P3
dP2 = P1 * P3
y0 = [P1, P2, P3]
variable_names = ["P1", "P2", "P3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    P1, P2, P3 = variables
    dP3: float = k1 * time
    dP1: float = -P1 * P3
    dP2: float = P1 * P3
    dP3dt: float = dP3
    dP1dt: float = dP1
    dP2dt: float = dP2
    return dP1dt, dP2dt, dP3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    P1, P2, P3 = variables
    dP3: float = k1 * time
    dP1: float = -P1 * P3
    dP2: float = P1 * P3
    return {
        "dP3": dP3,
        "dP1": dP1,
        "dP2": dP2,
    }
