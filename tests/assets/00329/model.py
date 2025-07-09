time: float = 0.0
k1: float = 1.50000000000000
compartment: float = 1.75000000000000
S3: float = 0.0
S1: float = 1.50000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
dS3 = 0.150000000000000
reaction1 = S1_conc * k1
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    S1_conc: float = S1 / compartment
    dS3: float = 0.150000000000000
    reaction1: float = S1_conc * k1
    dS3dt: float = compartment * dS3
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    dS3: float = 0.150000000000000
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "dS3": dS3,
        "reaction1": reaction1,
    }
