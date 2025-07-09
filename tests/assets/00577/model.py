time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
C: float = 1.00000000000000
S1_conc: float = 0.100000000000000
S2: float = 0.200000000000000
S3: float = 0.100000000000000

# Initial assignments
S1 = C * S1_conc
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S2, S3]
variable_names = ["S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S2, S3 = variables
    S1: float = C * S1_conc
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2
    return dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S2, S3 = variables
    S1: float = C * S1_conc
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "S1": S1,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
