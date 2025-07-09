time: float = 0.0

# Initial assignments
p1 = time % 2
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p1: float = time % 2
    return {
        "p1": p1,
    }
