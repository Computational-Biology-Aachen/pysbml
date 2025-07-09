import math

time: float = 0.0
k1: float = 0.600000000000000
k2: float = 0.130000000000000
compartment: float = math.nan
S1: float = 1.00000000000000
S2: float = 1.50000000000000
S3: float = 2.00000000000000
S4: float = 0.500000000000000

# Initial assignments
reaction1 = S1 * S2 * k1
reaction2 = S3 * S4 * k2
y0 = [S2, S3, S4]
variable_names = ["S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S2, S3, S4 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * S4 * k2
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2
    dS4dt: float = reaction1 - reaction2
    return dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S2, S3, S4 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * S4 * k2
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
