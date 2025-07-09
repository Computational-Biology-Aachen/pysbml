time: float = 0.0
C: float = 1.00000000000000
A: float = 1.00000000000000

# Initial assignments
J0 = 1
y0 = [A]
variable_names = ["A"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (A,) = variables
    J0: float = 1
    dAdt: float = 2.0 * J0
    return (dAdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (A,) = variables
    J0: float = 1
    return {
        "J0": J0,
    }
