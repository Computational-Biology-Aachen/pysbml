time: float = 0.0
p1: float = 5.00000000000000

# Initial assignments
p2 = time + 1
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p2: float = time + 1
    return {
        "p2": p2,
    }
