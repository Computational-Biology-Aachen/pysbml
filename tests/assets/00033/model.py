time: float = 0.0
compartment: float = 1.00000000000000
k1: float = 1.00000000000000
S1: float = 0.150000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
dk1 = 0.500000000000000
reaction1 = S1_conc * k1
y0 = [S1, S2, k1]
variable_names = ["S1", "S2", "k1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, k1 = variables
    S1_conc: float = S1 / compartment
    dk1: float = 0.500000000000000
    reaction1: float = S1_conc * k1
    dk1dt: float = dk1
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt, dk1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, k1 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    dk1: float = 0.500000000000000
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "dk1": dk1,
        "reaction1": reaction1,
    }
