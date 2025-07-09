time: float = 0.0
k2: float = 0.300000000000000
k3: float = -0.100000000000000
p1: float = 1.00000000000000
p2: float = 1.00000000000000
C: float = 1.25000000000000
k1: float = 1.70000000000000
S1: float = 1.00000000000000
S2: float = 2.00000000000000
S3: float = 1.00000000000000

# Initial assignments
dk1 = p1 * (k2 * p2 + k3)
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S1, S2, S3, k1]
variable_names = ["S1", "S2", "S3", "k1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, k1 = variables
    dk1: float = p1 * (k2 * p2 + k3)
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dk1dt: float = dk1
    dS1dt: float = -reaction1 + reaction2
    dS2dt: float = -reaction1 + reaction2
    dS3dt: float = reaction1 - reaction2
    return dS1dt, dS2dt, dS3dt, dk1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, k1 = variables
    dk1: float = p1 * (k2 * p2 + k3)
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "dk1": dk1,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
