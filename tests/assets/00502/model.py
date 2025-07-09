time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.00125000000000000
C: float = 1.00000000000000
S2: float = 0.0200000000000000
S3: float = 0.0100000000000000

# Initial assignments
S2_conc = S2 / C
S3_conc = S3 / C
reaction2 = S3_conc * k2
S1 = 2 * p1
S1_conc = S1 / C
reaction1 = S1_conc * S2_conc * k1
y0 = [S2, S3]
variable_names = ["S2", "S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S2, S3 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    dS2dt: float = -C * reaction1 + C * reaction2
    dS3dt: float = C * reaction1 - C * reaction2
    return dS2dt, dS3dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S2, S3 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * k2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
