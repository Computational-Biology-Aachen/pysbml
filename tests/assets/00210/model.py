time: float = 0.0
k1: float = 7500.00000000000
k2: float = 250.000000000000
compartment: float = 1.00000000000000

# Initial assignments
S1_amount = 0.001 * compartment
S2_amount = 0.0015 * compartment
S3_amount = 0.002 * compartment
S4_amount = 0.001 * compartment
S1 = S1_amount / compartment
S2 = S2_amount / compartment
S3 = S3_amount / compartment
S4 = S4_amount / compartment
reaction1 = S1 * S2 * k1
reaction2 = S3 * S4 * k2
y0 = [S1_amount, S2_amount, S3_amount, S4_amount]
variable_names = ["S1_amount", "S2_amount", "S3_amount", "S4_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount, S3_amount, S4_amount = variables
    S1: float = S1_amount / compartment
    S2: float = S2_amount / compartment
    S3: float = S3_amount / compartment
    S4: float = S4_amount / compartment
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * S4 * k2
    dS1_amountdt: float = -compartment * reaction1 + compartment * reaction2
    dS2_amountdt: float = -compartment * reaction1 + compartment * reaction2
    dS3_amountdt: float = compartment * reaction1 - compartment * reaction2
    dS4_amountdt: float = compartment * reaction1 - compartment * reaction2
    return dS1_amountdt, dS2_amountdt, dS3_amountdt, dS4_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount, S3_amount, S4_amount = variables
    S1: float = S1_amount / compartment
    S2: float = S2_amount / compartment
    S3: float = S3_amount / compartment
    S4: float = S4_amount / compartment
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * S4 * k2
    return {
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "S4": S4,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
