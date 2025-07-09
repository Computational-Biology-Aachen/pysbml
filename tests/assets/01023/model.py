time: float = 0.0
kf: float = 2.50000000000000
kr: float = 0.200000000000000
C: float = 1.00000000000000
S2: float = 0.500000000000000
S1: float = 1.00000000000000
S3: float = 0.0

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
reaction1 = -S1_conc * kf + S2_conc * S3_conc * kr
y0 = [S1, S3]
variable_names = ["S1", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S3 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = -S1_conc * kf + S2_conc * S3_conc * kr
    dS3dt: float = -C * reaction1
    dS1dt: float = reaction1
    return dS1dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S3 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = -S1_conc * kf + S2_conc * S3_conc * kr
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
