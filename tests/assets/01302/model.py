time: float = 0.0
p1: float = 0.0

# Initial assignments
J0 = 3
dp1 = J0
y0 = [p1]
variable_names = ["p1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (p1,) = variables
    J0: float = 3
    dp1: float = J0
    dp1dt: float = dp1
    return (dp1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (p1,) = variables
    J0: float = 3
    dp1: float = J0
    return {
        "J0": J0,
        "dp1": dp1,
    }
