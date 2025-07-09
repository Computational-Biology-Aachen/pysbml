time: float = 0.0
k1: float = 10.0000000000000
k2: float = 0.900000000000000
k3: float = 0.700000000000000
C: float = 1.00000000000000
S1: float = 0.200000000000000
S2: float = 0.200000000000000
S3: float = 0.0
S4: float = 0.0

# Initial assignments
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
reaction3 = S3 * k3
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    reaction3: float = S3 * k3
    dS1dt: float = -reaction1 + reaction2 + reaction3
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2 - reaction3
    dS4dt: float = reaction3
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    reaction3: float = S3 * k3
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
        "reaction3": reaction3,
    }
