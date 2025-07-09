time: float = 0.0
k1: float = 0.900000000000000
p1: float = 0.100000000000000
C: float = 1.00000000000000
S1: float = 1.50000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
dC = -C * p1
reaction1 = S1_conc * k1
y0 = [C, S1, S2]
variable_names = ["C", "S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C, S1, S2 = variables
    S1_conc: float = S1 / C
    dC: float = -C * p1
    reaction1: float = S1_conc * k1
    dCdt: float = dC
    dS1dt: float = -C * reaction1
    dS2dt: float = C * reaction1
    return dCdt, dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    C, S1, S2 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    dC: float = -C * p1
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "dC": dC,
        "reaction1": reaction1,
    }
