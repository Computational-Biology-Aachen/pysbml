time: float = 0.0
C: float = 1.00000000000000
S1: float = 1.00000000000000

# Initial assignments
J0 = False
J1 = True
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = False
    J1: float = True
    dS1dt: float = -J0 + J1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = False
    J1: float = True
    return {
        "J0": J0,
        "J1": J1,
    }
