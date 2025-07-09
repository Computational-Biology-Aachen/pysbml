time: float = 0.0
k1: float = 0.693000000000000
k2: float = 0.250000000000000
compartment: float = 1.00000000000000
S1: float = 0.00150000000000000
S3: float = 0.00150000000000000
S4: float = 0.00400000000000000
S2: float = 0.00200000000000000

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
S4_conc = S4 / compartment
dS3 = 0.0005 * k1
dS4 = -0.0005 * k2
reaction1 = S1_conc * k1
y0 = [S2, S3, S4]
variable_names = ["S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S2, S3, S4 = variables
    S1_conc: float = S1 / compartment
    dS3: float = 0.0005 * k1
    dS4: float = -0.0005 * k2
    reaction1: float = S1_conc * k1
    dS3dt: float = compartment * dS3
    dS4dt: float = dS4
    dS2dt: float = compartment * reaction1
    return dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S2, S3, S4 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    S4_conc: float = S4 / compartment
    dS3: float = 0.0005 * k1
    dS4: float = -0.0005 * k2
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "dS3": dS3,
        "dS4": dS4,
        "reaction1": reaction1,
    }
