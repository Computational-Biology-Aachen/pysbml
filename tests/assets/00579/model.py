time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
C: float = 1.00000000000000
S1_conc: float = 0.000100000000000000
S2_conc: float = 0.000200000000000000
S3_conc: float = 0.000100000000000000

# Initial assignments
S1 = C * S1_conc
S2 = C * S2_conc
S3 = C * S3_conc
reaction1 = C * S1 * S2 * k1
reaction2 = C * S3 * k2
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1: float = C * S1_conc
    S2: float = C * S2_conc
    S3: float = C * S3_conc
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1: float = C * S1_conc
    S2: float = C * S2_conc
    S3: float = C * S3_conc
    reaction1: float = C * S1 * S2 * k1
    reaction2: float = C * S3 * k2
    return {
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
