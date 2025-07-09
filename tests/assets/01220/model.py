time: float = 0.0
c: float = 2.00000000000000

# Initial assignments
dS1 = 1
S1 = 0.6 * c
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    dS1: float = 1
    dS1dt: float = dS1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    dS1: float = 1
    return {
        "dS1": dS1,
    }
