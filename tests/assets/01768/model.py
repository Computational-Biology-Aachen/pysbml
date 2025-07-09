time: float = 0.0
C: float = 2.00000000000000
J1_S1_stoich: float = 0.0100000000000000
k0: float = 0.0
S1_stoich: float = 2.00000000000000

# Initial assignments
dk0 = 2 * S1_stoich
J0 = 0.0100000000000000
J1 = J1_S1_stoich
S1_amount = 2.0 * C
S2_amount = 3.0 * C
S1 = S1_amount / C
S2 = S2_amount / C
y0 = [S1_amount, S2_amount, k0]
variable_names = ["S1_amount", "S2_amount", "k0"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount, k0 = variables
    dk0: float = 2 * S1_stoich
    J0: float = 0.0100000000000000
    J1: float = J1_S1_stoich
    dk0dt: float = dk0
    dS1_amountdt: float = -C * J0 * S1_stoich
    dS2_amountdt: float = C * J1
    return dS1_amountdt, dS2_amountdt, dk0dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount, k0 = variables
    S1: float = S1_amount / C
    S2: float = S2_amount / C
    dk0: float = 2 * S1_stoich
    J0: float = 0.0100000000000000
    J1: float = J1_S1_stoich
    return {
        "S1": S1,
        "S2": S2,
        "dk0": dk0,
        "J0": J0,
        "J1": J1,
    }
