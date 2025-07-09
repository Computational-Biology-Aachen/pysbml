time: float = 0.0
k1: float = 1.00000000000000
C: float = 0.900000000000000
S1: float = 0.00150000000000000
S2: float = 0.0

# Initial assignments
reaction1 = S1 * k1 * time
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    reaction1: float = S1 * k1 * time
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    reaction1: float = S1 * k1 * time
    return {
        "reaction1": reaction1,
    }
