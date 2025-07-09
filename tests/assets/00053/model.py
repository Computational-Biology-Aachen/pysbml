time: float = 0.0
k1: float = 0.750000000000000
k2: float = 0.250000000000000
p1: float = 0.100000000000000
C: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 1.50000000000000
S3: float = 2.00000000000000
S4: float = 1.00000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
dC = -C * p1
reaction1 = S1_conc * S2_conc * k1
reaction2 = S3_conc * S4_conc * k2
y0 = [C, S1, S2, S3, S4]
variable_names = ["C", "S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C, S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    dC: float = -C * p1
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * S4_conc * k2
    dCdt: float = dC
    dS1dt: float = -C * reaction1 + C * reaction2
    dS2dt: float = -C * reaction1 + C * reaction2
    dS3dt: float = C * reaction1 - C * reaction2
    dS4dt: float = C * reaction1 - C * reaction2
    return dCdt, dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    C, S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    dC: float = -C * p1
    reaction1: float = S1_conc * S2_conc * k1
    reaction2: float = S3_conc * S4_conc * k2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "dC": dC,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
