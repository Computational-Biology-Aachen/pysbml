time: float = 0.0
k1: float = 0.700000000000000
k2: float = 0.500000000000000
k3: float = 1.00000000000000
p1: float = 1.00000000000000
C: float = 1.00000000000000
S1: float = 0.100000000000000
S2: float = 0.0
S3: float = 0.0
S4: float = 0.0

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
S3_conc = S3 / C
S4_conc = S4 / C
reaction1 = S1_conc * k1 * time
reaction2 = S2_conc * k2 * time
reaction3 = S3_conc * k3 * time
generatedId_0 = 2 * p1
y0 = [S1, S2, S3, S4]
variable_names = ["S1", "S2", "S3", "S4"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    reaction1: float = S1_conc * k1 * time
    reaction2: float = S2_conc * k2 * time
    reaction3: float = S3_conc * k3 * time
    dS1dt: float = -C * reaction1
    dS2dt: float = -C * generatedId_0 * reaction2 + C * reaction1
    dS3dt: float = C * reaction2 - C * reaction3
    dS4dt: float = C * reaction3
    return dS1dt, dS2dt, dS3dt, dS4dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S3, S4 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    S3_conc: float = S3 / C
    S4_conc: float = S4 / C
    reaction1: float = S1_conc * k1 * time
    reaction2: float = S2_conc * k2 * time
    reaction3: float = S3_conc * k3 * time
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "S3_conc": S3_conc,
        "S4_conc": S4_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
        "reaction3": reaction3,
    }
