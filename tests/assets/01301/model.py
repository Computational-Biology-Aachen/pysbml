time: float = 0.0

# Initial assignments
J0 = 3
p1 = J0
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    J0: float = 3
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    J0: float = 3
    p1: float = J0
    return {
        "J0": J0,
        "p1": p1,
    }
