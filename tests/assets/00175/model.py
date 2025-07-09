time: float = 0.0
S1: float = 0.0

# Initial assignments
dS1 = 7
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    dS1: float = 7
    dS1dt: float = dS1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    dS1: float = 7
    return {
        "dS1": dS1,
    }
