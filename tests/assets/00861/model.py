time: float = 0.0
k1: float = 0.700000000000000
k2: float = 0.500000000000000
k3: float = 1.00000000000000
C: float = 1.78000000000000
S1: float = 0.100000000000000
S2: float = 0.0
S3: float = 0.0
S4: float = 0.0

# Initial assignments
reaction1 = S1 * k1 * time
reaction2 = S2 * k2 * time
reaction3 = S3 * k3 * time
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    reaction1: float = S1 * k1 * time
    reaction2: float = S2 * k2 * time
    reaction3: float = S3 * k3 * time
    dS1dt: float = -reaction1
    dS2dt: float = reaction1 - reaction2
    dS3dt: float = reaction2 - reaction3
    dS4dt: float = reaction3
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    reaction1: float = S1 * k1 * time
    reaction2: float = S2 * k2 * time
    reaction3: float = S3 * k3 * time
    return {
        "reaction1": reaction1,
        "reaction2": reaction2,
        "reaction3": reaction3,
    }
