time: float = 0.0
kf: float = 1.10000000000000
kr: float = 0.0900000000000000
C: float = 1.00000000000000
S3: float = 0.400000000000000
S1: float = 1.00000000000000
S2: float = 0.500000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
reaction1 = S1_conc * S2_conc * kf - S3_conc * kr
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S2_conc * kf - S3_conc * kr
    dS1dt: float = -reaction1
    dS2dt: float = -reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * S2_conc * kf - S3_conc * kr
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
