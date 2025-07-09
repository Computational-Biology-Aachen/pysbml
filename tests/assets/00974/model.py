time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000
Y: float = 0.0
Xref: float = 1.00000000000000
X: float = 0.0

# Initial assignments
Z = Xref
dY = Xref
J0 = k1
Q = Xref
y0 = [X, Y]
variable_names = ["X", "Y"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    X, Y = variables
    dY: float = Xref
    J0: float = k1
    dYdt: float = dY
    dXdt: float = J0 * Xref
    return dXdt, dYdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    X, Y = variables
    Z: float = Xref
    dY: float = Xref
    J0: float = k1
    return {
        "Z": Z,
        "dY": dY,
        "J0": J0,
    }
