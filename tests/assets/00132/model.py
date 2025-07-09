time: float = 0.0
compartment: float = 1.00000000000000
reaction1_k: float = 1.00000000000000
reaction2_k: float = 2.00000000000000
S1: float = 0.00300000000000000
S2: float = 0.0
S3: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
reaction1 = S1_conc * reaction1_k
reaction2 = S2_conc * reaction2_k
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * reaction1_k
    reaction2: float = S2_conc * reaction2_k
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1 - compartment * reaction2
    dS3dt: float = compartment * reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    reaction1: float = S1_conc * reaction1_k
    reaction2: float = S2_conc * reaction2_k
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
