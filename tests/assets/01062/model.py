time: float = 0.0
kf: float = 1.10000000000000
kr: float = 0.0900000000000000
C: float = 2.30000000000000
S1: float = 1.00000000000000
S2: float = 0.500000000000000
S3: float = 0.0

# Initial assignments
reaction1 = -S1 * S2 * kf + S3 * kr
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    reaction1: float = -S1 * S2 * kf + S3 * kr
    dS3dt: float = -reaction1
    dS1dt: float = reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    reaction1: float = -S1 * S2 * kf + S3 * kr
    return {
        "reaction1": reaction1,
    }
