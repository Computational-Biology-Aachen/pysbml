time: float = 0.0
k2: float = 0.200000000000000
k3: float = 0.300000000000000
C: float = 2.50000000000000
k1: float = 1.00000000000000
S1: float = 0.00150000000000000
S2: float = 0.0

# Initial assignments
dk1 = k2 + k3
reaction1 = S1 * k1
y0 = [S1, S2, k1]
variable_names = ["S1", "S2", "k1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, k1 = variables
    dk1: float = k2 + k3
    reaction1: float = S1 * k1
    dk1dt: float = dk1
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt, dk1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, k1 = variables
    dk1: float = k2 + k3
    reaction1: float = S1 * k1
    return {
        "dk1": dk1,
        "reaction1": reaction1,
    }
