time: float = 0.0
kf: float = 2.50000000000000
kr: float = 0.200000000000000
C: float = 1.00000000000000
S4: float = 0.500000000000000
S1: float = 1.00000000000000
S2: float = 0.500000000000000
S3: float = 0.0

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
dS4 = -0.5 * S1
reaction1 = -S1_conc * kf + S2_conc * S3_conc * kr
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    dS4: float = -0.5 * S1
    reaction1: float = -S1_conc * kf + S2_conc * S3_conc * kr
    dS4dt: float = C * dS4
    dS2dt: float = -C * reaction1
    dS3dt: float = -C * reaction1
    dS1dt: float = C * reaction1
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    dS4: float = -0.5 * S1
    reaction1: float = -S1_conc * kf + S2_conc * S3_conc * kr
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "dS4": dS4,
        "reaction1": reaction1,
    }
