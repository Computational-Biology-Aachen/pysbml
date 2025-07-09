time: float = 0.0
k1: float = 1.00000000000000
compartment: float = 10.0000000000000

# Initial assignments
S1_amount = 0.00015 * compartment
S2_amount = 0.0001 * compartment
S1 = S1_amount / compartment
S2 = S2_amount / compartment
reaction1 = S1 * k1
y0 = [S1_amount, S2_amount]
variable_names = ["S1_amount", "S2_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount = variables
    S1: float = S1_amount / compartment
    reaction1: float = S1 * k1
    dS1_amountdt: float = -compartment * reaction1
    dS2_amountdt: float = compartment * reaction1
    return dS1_amountdt, dS2_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount = variables
    S1: float = S1_amount / compartment
    S2: float = S2_amount / compartment
    reaction1: float = S1 * k1
    return {
        "S1": S1,
        "S2": S2,
        "reaction1": reaction1,
    }
