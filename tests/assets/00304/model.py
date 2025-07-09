time: float = 0.0
k1: float = 1.05000000000000
k2: float = 1.15000000000000
compartment: float = 1.00000000000000
S1: float = 0.100000000000000
S2: float = 0.150000000000000

# Initial assignments
S3 = S2 * k1
S1_conc = S1 / compartment
S2_conc = S2 / compartment
S3_conc = S3 / compartment
reaction1 = S1_conc * k2
y0 = [S2]
variable_names = ["S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S2,) = variables
    S3: float = S2 * k1
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * k2
    dS2dt: float = compartment * reaction1
    return (dS2dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S2,) = variables
    S3: float = S2 * k1
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    S3_conc: float = S3 / compartment
    reaction1: float = S1_conc * k2
    return {
        "S3": S3,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "reaction1": reaction1,
    }
