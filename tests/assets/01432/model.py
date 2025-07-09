time: float = 0.0
C: float = 1.00000000000000
A: float = 30.0000000000000
B: float = 1.00000000000000

# Initial assignments
J0 = 1
y0 = [A, B]
variable_names = ["A", "B"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    A, B = variables
    J0: float = 1
    dAdt: float = -3.0 * J0
    dBdt: float = 3.0 * J0
    return dAdt, dBdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    A, B = variables
    J0: float = 1
    return {
        "J0": J0,
    }
