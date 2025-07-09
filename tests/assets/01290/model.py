time: float = 0.0
p1: float = 1.00000000000000
p2: float = 2.00000000000000

# Initial assignments
dp1 = True
dp2 = False
y0 = [p1, p2]
variable_names = ["p1", "p2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    p1, p2 = variables
    dp1: float = True
    dp2: float = False
    dp1dt: float = dp1
    dp2dt: float = dp2
    return dp1dt, dp2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    p1, p2 = variables
    dp1: float = True
    dp2: float = False
    return {
        "dp1": dp1,
        "dp2": dp2,
    }
