time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
k3: float = 0.150000000000000
k4: float = 0.100000000000000
S1: float = 0.000100000000000000
S2: float = 0.000200000000000000
S3: float = 0.0
S4: float = 0.0

# Initial assignments
dS1 = -S1 * k1 + S2 * k2
dS3 = S2 * k3 - S3 * S4 * k4
dS4 = S2 * k3 - S3 * S4 * k4
y0 = [S1, S3, S4]
variable_names = ["S1", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S3, S4 = variables
    dS1: float = -S1 * k1 + S2 * k2
    dS3: float = S2 * k3 - S3 * S4 * k4
    dS4: float = S2 * k3 - S3 * S4 * k4
    dS1dt: float = dS1
    dS3dt: float = dS3
    dS4dt: float = dS4
    return dS1dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S3, S4 = variables
    dS1: float = -S1 * k1 + S2 * k2
    dS3: float = S2 * k3 - S3 * S4 * k4
    dS4: float = S2 * k3 - S3 * S4 * k4
    return {
        "dS1": dS1,
        "dS3": dS3,
        "dS4": dS4,
    }
