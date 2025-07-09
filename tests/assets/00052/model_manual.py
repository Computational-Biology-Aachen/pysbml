"""Manually created model to show best implementation."""

time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.100000000000000
C: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 2.00000000000000
S3: float = 1.00000000000000

# Initial assignments
y0 = [C, S1, S2, S3]
variable_names = ["C", "S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C, S1, S2, S3 = variables
    dC: float = -C * p1
    reaction1: float = C * S1 * S2 * k1
    reaction2: float = C * S3 * k2
    dCdt: float = dC
    dS1dt: float = -reaction1 / C**2 + reaction2 / C
    dS2dt: float = -reaction1 / C**2 + reaction2 / C
    dS3dt: float = reaction1 / C**2 - reaction2 / C
    return dCdt, dS1dt, dS2dt, dS3dt
