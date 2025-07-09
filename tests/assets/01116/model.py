time: float = 0.0

# Initial assignments
a = 1
b = (1) if (a > 0) else (2)
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    return {}
