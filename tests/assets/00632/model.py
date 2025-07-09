time: float = 0.0
p1: float = 75.0000000000000
p2: float = 0.0100000000000000
C: float = 1.00000000000000
S2: float = 0.150000000000000

# Initial assignments
k1 = p1 * p2
S2_conc = S2 / C
S1 = 0.133333333333333 * C * k1
S1_conc = S1 / C
reaction1 = S1_conc * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    k1: float = p1 * p2
    S1_conc: float = S1 / C
    reaction1: float = S1_conc * k1
    dS1dt: float = -C * reaction1
    dS2dt: float = C * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    k1: float = p1 * p2
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    reaction1: float = S1_conc * k1
    return {
        "k1": k1,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
