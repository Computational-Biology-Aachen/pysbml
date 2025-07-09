time: float = 0.0
k1: float = 0.750000000000000
k2: float = 2.70000000000000
C: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 1.50000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3 = C * S2 * k1
S3_conc = S3 / C
reaction1 = S1_conc * S3_conc * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S1_conc: float = S1 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S3_conc * k2
    dS1dt: float = -C * reaction1
    dS2dt: float = C * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S3_conc * k2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
