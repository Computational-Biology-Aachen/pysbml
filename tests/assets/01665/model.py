time: float = 0.0
p1: float = 1.00000000000000

# Initial assignments
dp1 = 6.02214179000000
y0 = [p1]
variable_names = ["p1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (p1,) = variables
    dp1: float = 6.02214179000000
    dp1dt: float = dp1
    return (dp1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (p1,) = variables
    dp1: float = 6.02214179000000
    return {
        "dp1": dp1,
    }
