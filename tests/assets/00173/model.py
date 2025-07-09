import math

import scipy.special

time: float = 0.0
p1: float = 4.00000000000000
p2: float = 25.0000000000000
S1: float = 1.00000000000000
S2: float = 0.0

# Initial assignments
dS1 = -scipy.special.factorial(math.ceil(S1 * p1)) / p2
dS2 = scipy.special.factorial(math.ceil(S1 * p1)) / p2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    dS1: float = -scipy.special.factorial(math.ceil(S1 * p1)) / p2
    dS2: float = scipy.special.factorial(math.ceil(S1 * p1)) / p2
    dS1dt: float = dS1
    dS2dt: float = dS2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    dS1: float = -scipy.special.factorial(math.ceil(S1 * p1)) / p2
    dS2: float = scipy.special.factorial(math.ceil(S1 * p1)) / p2
    return {
        "dS1": dS1,
        "dS2": dS2,
    }
