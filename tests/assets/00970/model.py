time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000
X: float = 0.0

# Initial assignments
Xref = 3
J0 = k1
y0 = [X]
variable_names = ["X"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (X,) = variables
    Xref: float = 3
    J0: float = k1
    dXdt: float = J0 * Xref
    return (dXdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (X,) = variables
    Xref: float = 3
    J0: float = k1
    return {
        "Xref": Xref,
        "J0": J0,
    }
