time: float = 0.0
C: float = 6.60000000000000
S2: float = 0.150000000000000

# Initial assignments
k1 = 0.750000000000000
S1 = 0.133333333333333 * k1
reaction1 = S1 * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    k1: float = 0.750000000000000
    reaction1: float = S1 * k1
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    k1: float = 0.750000000000000
    reaction1: float = S1 * k1
    return {
        "k1": k1,
        "reaction1": reaction1,
    }
