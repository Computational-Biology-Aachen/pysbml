time: float = 0.0
S1: float = 1.00000000000000
C: float = 1.00000000000000

# Initial assignments
dC = 1
y0 = [C]
variable_names = ["C"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (C,) = variables
    dC: float = 1
    dCdt: float = dC
    return (dCdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (C,) = variables
    dC: float = 1
    return {
        "dC": dC,
    }
