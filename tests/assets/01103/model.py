time: float = 0.0
default_compartment: float = 1.00000000000000
k1: float = 1.00000000000000
Xref: float = 1.00000000000000
X: float = 0.0

# Initial assignments
dXref = 0.0100000000000000
J0 = k1
y0 = [X, Xref]
variable_names = ["X", "Xref"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    X, Xref = variables
    dXref: float = 0.0100000000000000
    J0: float = k1
    dXrefdt: float = dXref
    dXdt: float = J0 * Xref
    return dXdt, dXrefdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    X, Xref = variables
    dXref: float = 0.0100000000000000
    J0: float = k1
    return {
        "dXref": dXref,
        "J0": J0,
    }
