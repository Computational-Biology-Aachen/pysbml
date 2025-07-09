import math

time: float = 0.0
k1: float = 1500.00000000000
k2: float = 50.0000000000000
k3: float = 1.50000000000000
compartment: float = math.nan
S1: float = 0.00100000000000000
S2: float = 0.00150000000000000
S3: float = 0.00100000000000000

# Initial assignments
S4 = S2 * k3
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dS1dt: float = -reaction1 + reaction2
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S4: float = S2 * k3
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "S4": S4,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
