time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.500000000000000
C: float = 1.00000000000000
reaction1_k1: float = 0.800000000000000
S3: float = 1.50000000000000
S4: float = 4.00000000000000
S2: float = 2.00000000000000

# Initial assignments
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
dS3 = S4 * k2
dS4 = -S4 * k2
S1 = 2.0 * C * k1
S1_conc = S1 / C
reaction1 = S1_conc * reaction1_k1
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    dS3: float = S4 * k2
    dS4: float = -S4 * k2
    reaction1: float = S1_conc * reaction1_k1
    dS3dt: float = C * dS3
    dS4dt: float = C * dS4
    dS1dt: float = -C * reaction1
    dS2dt: float = C * reaction1
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    dS3: float = S4 * k2
    dS4: float = -S4 * k2
    reaction1: float = S1_conc * reaction1_k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "dS3": dS3,
        "dS4": dS4,
        "reaction1": reaction1,
    }
