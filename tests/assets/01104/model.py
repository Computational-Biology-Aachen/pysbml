time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000

# Initial assignments
J0 = k1
X_amount = 1.0 * default_compartment
Y_amount = 1.0 * default_compartment
Xref = X_amount
X = X_amount / default_compartment
Y = Y_amount / default_compartment
J1 = Y * k1
y0 = [X_amount, Y_amount]
variable_names = ["X_amount", "Y_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    X_amount, Y_amount = variables
    Xref: float = X_amount
    Y: float = Y_amount / default_compartment
    J0: float = k1
    J1: float = Y * k1
    dX_amountdt: float = J0 * Xref * default_compartment
    dY_amountdt: float = J1 * default_compartment
    return dX_amountdt, dY_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    X_amount, Y_amount = variables
    Xref: float = X_amount
    X: float = X_amount / default_compartment
    Y: float = Y_amount / default_compartment
    J0: float = k1
    J1: float = Y * k1
    return {
        "Xref": Xref,
        "X": X,
        "Y": Y,
        "J0": J0,
        "J1": J1,
    }
