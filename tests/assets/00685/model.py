time: float = 0.0
k1: float = 1.50000000000000
k2: float = 5.00000000000000
C: float = 2.50000000000000
S1: float = 1.00000000000000
S2: float = 0.500000000000000

# Initial assignments
S3 = S2 * k1
S3_conc = S3 / C
reaction1 = S1 * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S3: float = S2 * k1
    reaction1: float = S1 * k2
    dS1dt: float = -reaction1
    dS2dt: float = reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S3: float = S2 * k1
    S3_conc: float = S3 / C
    reaction1: float = S1 * k2
    return {
        "S3": S3,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
