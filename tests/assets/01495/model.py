import math

time: float = 0.0

# Initial assignments
p1 = math.floor((1 / 2) * time)
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p1: float = math.floor((1 / 2) * time)
    return {
        "p1": p1,
    }
