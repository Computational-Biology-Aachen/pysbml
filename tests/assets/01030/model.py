time: float = 0.0
kf: float = 0.900000000000000
kr: float = 0.0750000000000000
C: float = 1.00000000000000
reaction1_kf: float = 0.800000000000000
reaction1_kr: float = 0.0600000000000000
S1: float = 1.00000000000000
S2: float = 0.0
S3: float = 1.50000000000000
S4: float = 0.500000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
reaction1 = -S1_conc * reaction1_kf + S2_conc * reaction1_kr
reaction2 = -S3_conc * kf + S4_conc * kr
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    reaction1: float = -S1_conc * reaction1_kf + S2_conc * reaction1_kr
    reaction2: float = -S3_conc * kf + S4_conc * kr
    dS2dt: float = -C * reaction1
    dS1dt: float = C * reaction1
    dS4dt: float = -C * reaction2
    dS3dt: float = C * reaction2
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    reaction1: float = -S1_conc * reaction1_kf + S2_conc * reaction1_kr
    reaction2: float = -S3_conc * kf + S4_conc * kr
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
