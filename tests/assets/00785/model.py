time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.250000000000000

# Initial assignments
C = 2 * p1
S1_amount = 0.001 * C
S2_amount = 0.002 * C
S3_amount = 0.001 * C
S1 = S1_amount / C
S2 = S2_amount / C
S3 = S3_amount / C
reaction1 = C * S1 * S2 * k1
reaction2 = C * S3 * k2
y0 = [S1_amount, S2_amount, S3_amount]
variable_names = ["S1_amount", "S2_amount", "S3_amount"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1_amount, S2_amount, S3_amount = variables
    S1: float = S1_amount / C
    S2: float = S2_amount / C
    S3: float = S3_amount / C
    reaction1: float = C * S1 * S2 * k1
    reaction2: float = C * S3 * k2
    dS1_amountdt: float = -reaction1 + reaction2
    dS2_amountdt: float = -reaction1 + reaction2
    dS3_amountdt: float = reaction1 - reaction2
    return dS1_amountdt, dS2_amountdt, dS3_amountdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_amount, S2_amount, S3_amount = variables
    S1: float = S1_amount / C
    S2: float = S2_amount / C
    S3: float = S3_amount / C
    reaction1: float = C * S1 * S2 * k1
    reaction2: float = C * S3 * k2
    return {
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
