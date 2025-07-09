time: float = 0.0
k1: float = 0.700000000000000
k2: float = 0.500000000000000
k3: float = 1.00000000000000
S1: float = 0.0100000000000000
S2: float = 0.0
S3: float = 0.0
S4: float = 0.0

# Initial assignments
dS1 = -S1 * k1
dS2 = S1 * k1 - S2 * k2
dS3 = S2 * k2 - S3 * k3
dS4 = S3 * k3
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    dS1: float = -S1 * k1
    dS2: float = S1 * k1 - S2 * k2
    dS3: float = S2 * k2 - S3 * k3
    dS4: float = S3 * k3
    dS1dt: float = dS1
    dS2dt: float = dS2
    dS3dt: float = dS3
    dS4dt: float = dS4
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    dS1: float = -S1 * k1
    dS2: float = S1 * k1 - S2 * k2
    dS3: float = S2 * k2 - S3 * k3
    dS4: float = S3 * k3
    return {
        "dS1": dS1,
        "dS2": dS2,
        "dS3": dS3,
        "dS4": dS4,
    }
