time: float = 0.0
C: float = 2.00000000000000
S1_stoich: float = 2.00000000000000

# Initial assignments
J0 = 0.0100000000000000
k0 = 2 * S1_stoich
S1_amount = 2.0 * C
S2_amount = 3.0 * C
S1 = S1_amount / C
S2 = S2_amount / C
y0 = [S1_amount, S2_amount]
variable_names = ["S1_amount", "S2_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount = variables
    J0: float = 0.0100000000000000
    dS1_amountdt: float = -C * J0 * S1_stoich
    dS2_amountdt: float = C * J0
    return dS1_amountdt, dS2_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount = variables
    S1: float = S1_amount / C
    S2: float = S2_amount / C
    J0: float = 0.0100000000000000
    return {
        "S1": S1,
        "S2": S2,
        "J0": J0,
    }
