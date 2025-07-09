import math

time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
compartment: float = math.nan
S3: float = 1.50000000000000
S4: float = 4.00000000000000
S1: float = 1.50000000000000
S2: float = 2.00000000000000

# Initial assignments
dS3 = 0.5 * k1
dS4 = -0.5 * k2
reaction1 = S1 * k1
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    dS3: float = 0.5 * k1
    dS4: float = -0.5 * k2
    reaction1: float = S1 * k1
    dS3dt: float = dS3
    dS4dt: float = dS4
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    dS3: float = 0.5 * k1
    dS4: float = -0.5 * k2
    reaction1: float = S1 * k1
    return {
        "dS3": dS3,
        "dS4": dS4,
        "reaction1": reaction1,
    }
