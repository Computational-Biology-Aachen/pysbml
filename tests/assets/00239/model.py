import math

time: float = 0.0
k1: float = 1.02300000000000
compartment: float = math.nan
S2: float = 0.0
S1: float = 0.150000000000000

# Initial assignments
reaction1 = S1 * k1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    reaction1: float = S1 * k1
    dS1dt: float = -reaction1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    reaction1: float = S1 * k1
    return {
        "reaction1": reaction1,
    }
