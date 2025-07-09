time: float = 0.0
k1: float = 0.500000000000000
k2: float = 50.0000000000000

# Initial assignments
C = (1 / 50) * k2
S1_amount = 1.0 * C
S2_amount = 1.5 * C
S1 = S1_amount / C
S2 = S2_amount / C
reaction1 = C * S1 * k1
y0 = [S1_amount, S2_amount]
variable_names = ["S1_amount", "S2_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount = variables
    S1: float = S1_amount / C
    reaction1: float = C * S1 * k1
    dS1_amountdt: float = -reaction1
    dS2_amountdt: float = reaction1
    return dS1_amountdt, dS2_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount = variables
    S1: float = S1_amount / C
    S2: float = S2_amount / C
    reaction1: float = C * S1 * k1
    return {
        "S1": S1,
        "S2": S2,
        "reaction1": reaction1,
    }
