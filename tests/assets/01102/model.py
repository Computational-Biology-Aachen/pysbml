time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000
X: float = 0.0

# Initial assignments
X_conc = X / default_compartment
J0 = k1
Xref = 3
y0 = [X]
variable_names = ["X"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (X,) = variables
    J0: float = k1
    dXdt: float = J0 * Xref * default_compartment
    return (dXdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (X,) = variables
    X_conc: float = X / default_compartment
    J0: float = k1
    return {
        "X_conc": X_conc,
        "J0": J0,
    }
