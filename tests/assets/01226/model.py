time: float = 0.0
c: float = 1.00000000000000
p1: float = 0.0
S1: float = 0.0

# Initial assignments
J0 = 1
dp1 = J0
y0 = [S1, p1]
variable_names = ["S1", "p1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, p1 = variables
    J0: float = 1
    dp1: float = J0
    dp1dt: float = dp1
    dS1dt: float = J0
    return dS1dt, dp1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, p1 = variables
    J0: float = 1
    dp1: float = J0
    return {
        "J0": J0,
        "dp1": dp1,
    }
