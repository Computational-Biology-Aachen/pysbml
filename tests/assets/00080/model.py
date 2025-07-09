time: float = 0.0
k1: float = 0.100000000000000
k2: float = 0.150000000000000
p1: float = 2.50000000000000
compartment: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 0.0
S3: float = 0.0

# Initial assignments
S4 = S3 / (p1 + 1)
S5 = S4 * p1
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
S4_conc = S4 / compartment
S5_conc = S5 / compartment
reaction1 = S1_conc * k1
reaction2 = S5_conc * k2
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    S4: float = S3 / (p1 + 1)
    S5: float = S4 * p1
    S1_conc: float = S1 / compartment
    S5_conc: float = S5 / compartment
    reaction1: float = S1_conc * k1
    reaction2: float = S5_conc * k2
    dS1dt: float = -compartment * reaction1
    dS3dt: float = compartment * reaction1 - compartment * reaction2
    dS2dt: float = compartment * reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S4: float = S3 / (p1 + 1)
    S5: float = S4 * p1
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    S4_conc: float = S4 / compartment
    S5_conc: float = S5 / compartment
    reaction1: float = S1_conc * k1
    reaction2: float = S5_conc * k2
    return {
        "S4": S4,
        "S5": S5,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "S5_conc": S5_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
