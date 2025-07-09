time: float = 0.0
compartment: float = 1.00000000000000
S1: float = 0.100000000000000
S2: float = 0.150000000000000

# Initial assignments
k1 = 0.750000000000000
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = S1_conc * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    k1: float = 0.750000000000000
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * k1
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    k1: float = 0.750000000000000
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * k1
    return {
        "k1": k1,
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
