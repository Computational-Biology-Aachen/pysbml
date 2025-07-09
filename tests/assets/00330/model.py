time: float = 0.0
k1: float = 75.0000000000000
k2: float = 0.0250000000000000
compartment: float = 0.930000000000000
S4: float = 0.0100000000000000
S1: float = 0.0150000000000000
S2: float = 0.0200000000000000
S3: float = 0.0150000000000000

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
S4_conc = S4 / compartment
dS4 = 0.0100000000000000
reaction1 = S1_conc * S2_conc * k1
reaction2 = S3_conc * k2
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    dS4: float = 0.0100000000000000
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    dS4dt: float = compartment * dS4
    dS1dt: float = -compartment * reaction1 + compartment * reaction2
    dS2dt: float = -compartment * reaction1 + compartment * reaction2
    dS3dt: float = compartment * reaction1 - compartment * reaction2
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    S4_conc: float = S4 / compartment
    dS4: float = 0.0100000000000000
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "dS4": dS4,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
