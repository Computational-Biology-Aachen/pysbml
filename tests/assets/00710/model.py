time: float = 0.0
k1: float = 1.50000000000000
C: float = 1.75000000000000
S3: float = 0.0
S1: float = 1.50000000000000
S2: float = 0.0

# Initial assignments
dS3 = 0.150000000000000
reaction1 = S1 * k1
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    dS3: float = 0.150000000000000
    reaction1: float = S1 * k1
    dS3dt: float = dS3
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    dS3: float = 0.150000000000000
    reaction1: float = S1 * k1
    return {
        "dS3": dS3,
        "reaction1": reaction1,
    }
