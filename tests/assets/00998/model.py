time: float = 0.0
comp: float = 5.00000000000000

# Initial assignments
S1 = 1.0 * comp
J0 = 10 / S1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = 10 / S1
    dS1dt: float = J0
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = 10 / S1
    return {
        "J0": J0,
    }
