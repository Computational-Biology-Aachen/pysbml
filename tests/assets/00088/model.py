time: float = 0.0
k1: float = 1.50000000000000
k2: float = 50.0000000000000
compartment: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 1.50000000000000

# Initial assignments
S3 = S1 * k1
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
reaction1 = S1_conc * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S3: float = S1 * k1
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * k2
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S3: float = S1 * k1
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    reaction1: float = S1_conc * k2
    return {
        "S3": S3,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
