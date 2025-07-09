time: float = 0.0
C: float = 1.00000000000000
reaction1_k: float = 1.00000000000000
reaction2_k: float = 2.00000000000000
S1: float = 0.00300000000000000
S2: float = 0.0
S3: float = 0.0

# Initial assignments
reaction1 = S1 * reaction1_k
reaction2 = S2 * reaction2_k
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    reaction1: float = S1 * reaction1_k
    reaction2: float = S2 * reaction2_k
    dS1dt: float = -reaction1
    dS2dt: float = reaction1 - reaction2
    dS3dt: float = reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    reaction1: float = S1 * reaction1_k
    reaction2: float = S2 * reaction2_k
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
