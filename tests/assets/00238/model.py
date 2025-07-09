import math

time: float = 0.0
k1: float = 1.02300000000000
compartment: float = math.nan
S1: float = 0.150000000000000
S2: float = 0.0

# Initial assignments
reaction1 = S1 * k1
y0 = [S2]
variable_names = ["S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S2,) = variables
    reaction1: float = S1 * k1
    dS2dt: float = reaction1
    return (dS2dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S2,) = variables
    reaction1: float = S1 * k1
    return {
        "reaction1": reaction1,
    }
