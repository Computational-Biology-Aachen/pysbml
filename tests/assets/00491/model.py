import math

time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.0125000000000000
C: float = math.nan
S2: float = 0.200000000000000
S3: float = 0.100000000000000

# Initial assignments
reaction2 = S3 * k2
S1 = 2 * p1
reaction1 = S1 * S2 * k1
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
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
