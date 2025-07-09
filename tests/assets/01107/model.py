time: float = 0.0
k1: float = 1.00000000000000
default_compartment: float = 1.00000000000000

# Initial assignments
J0 = k1
X_amount = 1.0 * default_compartment
p1 = X_amount
X = X_amount / default_compartment
Xref = p1
y0 = [X_amount]
variable_names = ["X_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (X_amount,) = variables
    p1: float = X_amount
    J0: float = k1
    Xref: float = p1
    dX_amountdt: float = J0 * Xref * default_compartment
    return (dX_amountdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (X_amount,) = variables
    p1: float = X_amount
    X: float = X_amount / default_compartment
    J0: float = k1
    Xref: float = p1
    return {
        "p1": p1,
        "X": X,
        "J0": J0,
        "Xref": Xref,
    }
