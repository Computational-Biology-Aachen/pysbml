time: float = 0.0
k1: float = 0.693000000000000
k2: float = 0.250000000000000
compartment: float = 1.00000000000000

# Initial assignments
dS3 = 0.0005 * k1
dS4 = -0.0005 * k2
S1_amount = 0.0015 * compartment
S2_amount = 0.002 * compartment
S3_amount = 0.0015 * compartment
S4_amount = 0.004 * compartment
S1 = S1_amount / compartment
S2 = S2_amount / compartment
S3 = S3_amount / compartment
S4 = S4_amount / compartment
reaction1 = S1 * k1
y0 = [S1_amount, S2_amount, S3_amount, S4_amount]
variable_names = ["S1_amount", "S2_amount", "S3_amount", "S4_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount, S3_amount, S4_amount = variables
    S1: float = S1_amount / compartment
    dS3: float = 0.0005 * k1
    dS4: float = -0.0005 * k2
    reaction1: float = S1 * k1
    dS3_amountdt: float = compartment * dS3
    dS4_amountdt: float = compartment * dS4
    dS1_amountdt: float = -compartment * reaction1
    dS2_amountdt: float = compartment * reaction1
    return dS1_amountdt, dS2_amountdt, dS3_amountdt, dS4_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount, S3_amount, S4_amount = variables
    S1: float = S1_amount / compartment
    S2: float = S2_amount / compartment
    S3: float = S3_amount / compartment
    S4: float = S4_amount / compartment
    dS3: float = 0.0005 * k1
    dS4: float = -0.0005 * k2
    reaction1: float = S1 * k1
    return {
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "S4": S4,
        "dS3": dS3,
        "dS4": dS4,
        "reaction1": reaction1,
    }
