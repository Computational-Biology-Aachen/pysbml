time: float = 0.0
k1: float = 0.900000000000000
p1: float = 0.100000000000000
p2: float = 1.50000000000000
S1: float = 1.50000000000000
S2: float = 0.0

# Initial assignments
C = p1 * p2
S1_conc = S1 / C
S2_conc = S2 / C
dp2 = 0.100000000000000
reaction1 = S1_conc * k1
y0 = [S1, S2, p2]
variable_names = ["S1", "S2", "p2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, p2 = variables
    C: float = p1 * p2
    S1_conc: float = S1 / C
    dp2: float = 0.100000000000000
    reaction1: float = S1_conc * k1
    dp2dt: float = dp2
    dS1dt: float = -C * reaction1
    dS2dt: float = C * reaction1
    return dS1dt, dS2dt, dp2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, p2 = variables
    C: float = p1 * p2
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    dp2: float = 0.100000000000000
    reaction1: float = S1_conc * k1
    return {
        "C": C,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "dp2": dp2,
        "reaction1": reaction1,
    }
