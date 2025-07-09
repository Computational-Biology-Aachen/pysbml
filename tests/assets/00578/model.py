time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
C: float = 1.00000000000000
S1_conc: float = 0.0100000000000000
S2_conc: float = 0.0200000000000000
S3: float = 0.0100000000000000

# Initial assignments
S1 = C * S1_conc
S2 = C * S2_conc
reaction1 = S1 * S2 * k1
reaction2 = S3 * k2
y0 = [S3]
variable_names = ["S3"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S3,) = variables
    S1: float = C * S1_conc
    S2: float = C * S2_conc
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    dS3dt: float = reaction1 - reaction2
    return (dS3dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S3,) = variables
    S1: float = C * S1_conc
    S2: float = C * S2_conc
    reaction1: float = S1 * S2 * k1
    reaction2: float = S3 * k2
    return {
        "S1": S1,
        "S2": S2,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
