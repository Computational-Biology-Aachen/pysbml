import math

time: float = 0.0
k1: float = 9.80000000000000
k2: float = 0.570000000000000
compartment: float = math.nan
S3: float = 1.00000000000000e-6
S1: float = 1.00000000000000e-6
S2: float = 2.00000000000000e-6

# Initial assignments
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dS1dt: float = -reaction1 + reaction2
    dS2dt: float = -reaction1 + reaction2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
