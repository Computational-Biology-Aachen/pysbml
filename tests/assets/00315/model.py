time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.100000000000000
p2: float = 1.00000000000000
S1: float = 0.100000000000000
S2: float = 0.200000000000000
S3: float = 0.100000000000000

# Initial assignments
C = p1 * p2
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
dp2 = 0.100000000000000
reaction1 = S1_conc * S2_conc * k1
reaction2 = S3_conc * k2
y0 = [S1, S2, S3, p2]
variable_names = ["S1", "S2", "S3", "p2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, p2 = variables
    C: float = p1 * p2
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    dp2: float = 0.100000000000000
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    dp2dt: float = dp2
    dS1dt: float = -C * reaction1 + C * reaction2
    dS2dt: float = -C * reaction1 + C * reaction2
    dS3dt: float = C * reaction1 - C * reaction2
    return dS1dt, dS2dt, dS3dt, dp2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, p2 = variables
    C: float = p1 * p2
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    dp2: float = 0.100000000000000
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    return {
        "C": C,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "dp2": dp2,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
