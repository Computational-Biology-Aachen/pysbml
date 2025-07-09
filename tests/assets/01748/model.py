time: float = 0.0
C: float = 2.00000000000000

# Initial assignments
S1_stoich = 0.1 * time
J0 = 0.100000000000000
S1_amount = 2.0 * C
S1 = S1_amount / C
y0 = [S1_amount]
variable_names = ["S1_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1_amount,) = variables
    S1_stoich: float = 0.1 * time
    J0: float = 0.100000000000000
    dS1_amountdt: float = C * J0 * S1_stoich
    return (dS1_amountdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1_amount,) = variables
    S1_stoich: float = 0.1 * time
    S1: float = S1_amount / C
    J0: float = 0.100000000000000
    return {
        "S1_stoich": S1_stoich,
        "S1": S1,
        "J0": J0,
    }
