time: float = 0.0
k1: float = 0.750000000000000
k2: float = 7.50000000000000
k3: float = 0.750000000000000
k4: float = 0.500000000000000
compartment: float = 1.00000000000000
compartment1: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 1.00000000000000
S3: float = 0.0
S4: float = 0.100000000000000

# Initial assignments
S5 = S2 * k4
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment1
S4_conc = S4 / compartment1
S5_conc = S5 / compartment1
reaction1 = S1_conc * S2_conc * k1
reaction2 = k2 * (S2_conc - S3_conc)
reaction3 = S3_conc * S4_conc * compartment * k3
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S5: float = S2 * k4
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment1
    S4_conc: float = S4 / compartment1
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = k2 * (S2_conc - S3_conc)
    reaction3: float = S3_conc * S4_conc * compartment * k3
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1 - compartment * reaction2
    dS3dt: float = compartment1 * reaction2 - compartment1 * reaction3
    dS4dt: float = compartment1 * reaction3
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S5: float = S2 * k4
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment1
    S4_conc: float = S4 / compartment1
    S5_conc: float = S5 / compartment1
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = k2 * (S2_conc - S3_conc)
    reaction3: float = S3_conc * S4_conc * compartment * k3
    return {
        "S5": S5,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "S5_conc": S5_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
        "reaction3": reaction3,
    }
