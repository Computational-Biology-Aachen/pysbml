time: float = 0.0

# Initial assignments
p1 = max(-5, -time)
p2 = time
p3 = max(5, time)
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p1: float = max(-5, -time)
    p2: float = time
    p3: float = max(5, time)
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }
