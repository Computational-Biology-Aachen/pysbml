time: float = 0.0
k1: float = 0.00500000000000000
k2: float = 5.00000000000000
k3: float = 1.50000000000000
compartment: float = 1.00000000000000
S3: float = 1.00000000000000e-5
S1: float = 1.00000000000000e-5
S2: float = 1.50000000000000e-5

# Initial assignments
S4 = S2 * k3
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
S4_conc = S4 / compartment
reaction1 = S1_conc * S2_conc * k1
reaction2 = S3_conc * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S4: float = S2 * k3
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    dS1dt: float = -compartment * reaction1 + reaction2
    dS2dt: float = -compartment * reaction1 + reaction2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S4: float = S2 * k3
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    S4_conc: float = S4 / compartment
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    return {
        "S4": S4,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
