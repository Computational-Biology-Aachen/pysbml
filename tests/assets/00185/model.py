time: float = 0.0
k1: float = 0.750000000000000
k2: float = 50.0000000000000

# Initial assignments
S1 = 1.33333333333333 * k1
S2 = 3.0e-17 * k2
S3 = S2 * k1
dS1 = -S1 * k2
dS2 = S1 * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    dS1: float = -S1 * k2
    dS2: float = S1 * k2
    dS1dt: float = dS1
    dS2dt: float = dS2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S3: float = S2 * k1
    dS1: float = -S1 * k2
    dS2: float = S1 * k2
    return {
        "S3": S3,
        "dS1": dS1,
        "dS2": dS2,
    }
