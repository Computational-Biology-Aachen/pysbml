time: float = 0.0

# Initial assignments
S1 = 7
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1: float = 7
    return {
        "S1": S1,
    }
