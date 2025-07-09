time: float = 0.0
k: float = 1.50000000000000
C: float = 1.00000000000000
reaction1_k: float = 15000.0000000000
reaction2_k: float = 5.00000000000000
S1: float = 1.00000000000000e-5
S2: float = 1.50000000000000e-5
S3: float = 1.00000000000000e-5

# Initial assignments
S4 = S2 * k
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
reaction1 = S1_conc * S2_conc * reaction1_k
reaction2 = S3_conc * reaction2_k
y0 = [S1, S2, S3]
variable_names = ["S1", "S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3 = variables
    S4: float = S2 * k
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S2_conc * reaction1_k
    reaction2: float = S3_conc * reaction2_k
    dS1dt: float = -C * reaction1 + C * reaction2
    dS2dt: float = -C * reaction1 + C * reaction2
    dS3dt: float = C * reaction1 - C * reaction2
    return dS1dt, dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3 = variables
    S4: float = S2 * k
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    reaction1: float = S1_conc * S2_conc * reaction1_k
    reaction2: float = S3_conc * reaction2_k
    return {
        "S4": S4,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
