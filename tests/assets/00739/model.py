time: float = 0.0
k1: float = 0.100000000000000
k2: float = 0.150000000000000
p1: float = 2.50000000000000
C: float = 2.50000000000000
S1: float = 1.00000000000000
S2: float = 0.0
S3: float = 0.0

# Initial assignments
S4 = S3 / (p1 + 1)
S5 = S4 * p1
reaction1 = S1 * k1
reaction2 = S5 * k2
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    S4: float = S3 / (p1 + 1)
    S5: float = S4 * p1
    reaction1: float = S1 * k1
    reaction2: float = S5 * k2
    dS1dt: float = -reaction1
    dS3dt: float = reaction1 - reaction2
    dS2dt: float = reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S4: float = S3 / (p1 + 1)
    S5: float = S4 * p1
    reaction1: float = S1 * k1
    reaction2: float = S5 * k2
    return {
        "S4": S4,
        "S5": S5,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
