import math

time: float = 0.0
k1: float = 69.0000000000000
k2: float = 0.0230000000000000
compartment: float = math.nan
S4: float = 0.0100000000000000
S1: float = 0.0150000000000000
S2: float = 0.0200000000000000
S3: float = 0.0400000000000000

# Initial assignments
dS4 = 0.0100000000000000
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    dS4: float = 0.0100000000000000
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dS4dt: float = dS4
    dS1dt: float = -reaction1 + reaction2
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    dS4: float = 0.0100000000000000
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "dS4": dS4,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
