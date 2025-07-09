time: float = 0.0
C: float = 1.00000000000000
A: float = 1.00000000000000

# Initial assignments
dA1_sr2 = -1
dA1_sr = 1
J0 = 1
A1_sr = 2
A1_sr2 = 1
y0 = [A, A1_sr, A1_sr2]
variable_names = ["A", "A1_sr", "A1_sr2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    A, A1_sr, A1_sr2 = variables
    dA1_sr2: float = -1
    dA1_sr: float = 1
    J0: float = 1
    dA1_sr2dt: float = dA1_sr2
    dA1_srdt: float = dA1_sr
    dAdt: float = J0 * (A1_sr + A1_sr2)
    return dAdt, dA1_srdt, dA1_sr2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    A, A1_sr, A1_sr2 = variables
    dA1_sr2: float = -1
    dA1_sr: float = 1
    J0: float = 1
    return {
        "dA1_sr2": dA1_sr2,
        "dA1_sr": dA1_sr,
        "J0": J0,
    }
