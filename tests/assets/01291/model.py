time: float = 0.0

# Initial assignments
p1 = True
p2 = False
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p1: float = True
    p2: float = False
    return {
        "p1": p1,
        "p2": p2,
    }
