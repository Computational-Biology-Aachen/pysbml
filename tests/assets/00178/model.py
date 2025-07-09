time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
S1: float = 0.100000000000000
S2: float = 0.200000000000000
S3: float = 0.100000000000000

# Initial assignments
dS3 = S1 * S2 * k1 - S3 * k2
dS1 = -S1 * S2 * k1 + S3 * k2
dS2 = -S1 * S2 * k1 + S3 * k2
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    dS3: float = S1 * S2 * k1 - S3 * k2
    dS1: float = -S1 * S2 * k1 + S3 * k2
    dS2: float = -S1 * S2 * k1 + S3 * k2
    dS3dt: float = dS3
    dS1dt: float = dS1
    dS2dt: float = dS2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    dS3: float = S1 * S2 * k1 - S3 * k2
    dS1: float = -S1 * S2 * k1 + S3 * k2
    dS2: float = -S1 * S2 * k1 + S3 * k2
    return {
        "dS3": dS3,
        "dS1": dS1,
        "dS2": dS2,
    }
