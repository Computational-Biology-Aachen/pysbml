time: float = 0.0
C1: float = 1.00000000000000
S1_sr: float = 2.00000000000000
S1: float = 0.0

# Initial assignments
J0 = 1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = 1
    dS1dt: float = J0 * S1_sr
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = 1
    return {
        "J0": J0,
    }
