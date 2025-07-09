time: float = 0.0
C: float = 1.00000000000000
A: float = 1.00000000000000
B: float = 1.00000000000000

# Initial assignments
J0 = -1
A_sr1 = 2
A_sr2 = 1
B_sr1 = 2
B_sr2 = 1
y0 = [A, B]
variable_names = ["A", "B"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    A, B = variables
    J0: float = -1
    dAdt: float = J0 * (-A_sr1 + A_sr2)
    dBdt: float = J0 * (-B_sr1 + B_sr2)
    return dAdt, dBdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    A, B = variables
    J0: float = -1
    return {
        "J0": J0,
    }
