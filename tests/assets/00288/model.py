time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
k4: float = 0.500000000000000
compartment: float = 1.00000000000000
compartment1: float = 1.00000000000000
S1: float = 1.00000000000000
S3: float = 0.0

# Initial assignments
S5 = S3 * k4
S1_conc = S1 / compartment
S3_conc = S3 / compartment1
S5_conc = S5 / compartment1
reaction1 = S1_conc * k1
reaction2 = k2 * (-S1_conc + S3_conc)
y0 = [S1, S3]
variable_names = ["S1", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S3 = variables
    S5: float = S3 * k4
    S1_conc: float = S1 / compartment
    S3_conc: float = S3 / compartment1
    reaction1: float = S1_conc * k1
    reaction2: float = k2 * (-S1_conc + S3_conc)
    dS1dt: float = -compartment * reaction1 + compartment * reaction2
    dS3dt: float = compartment1 * reaction1 - compartment1 * reaction2
    return dS1dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S3 = variables
    S5: float = S3 * k4
    S1_conc: float = S1 / compartment
    S3_conc: float = S3 / compartment1
    S5_conc: float = S5 / compartment1
    reaction1: float = S1_conc * k1
    reaction2: float = k2 * (-S1_conc + S3_conc)
    return {
        "S5": S5,
        "S1_conc": S1_conc,
        "S3_conc": S3_conc,
        "S5_conc": S5_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
