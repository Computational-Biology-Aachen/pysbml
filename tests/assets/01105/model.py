time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000
p1: float = 1.00000000000000
X: float = 1.00000000000000

# Initial assignments
Xref = p1
dp1 = 1
J0 = k1
y0 = [X, p1]
variable_names = ["X", "p1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    X, p1 = variables
    Xref: float = p1
    dp1: float = 1
    J0: float = k1
    dp1dt: float = dp1
    dXdt: float = J0 * Xref
    return dXdt, dp1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    X, p1 = variables
    Xref: float = p1
    dp1: float = 1
    J0: float = k1
    return {
        "Xref": Xref,
        "dp1": dp1,
        "J0": J0,
    }
