time: float = 0.0
k1: float = 0.350000000000000
k2: float = 180.000000000000
compartment: float = 1.00000000000000
S1: float = 0.0150000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = S1_conc * k1
reaction2 = S2_conc**2 * k2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * k1
    reaction2: float = S2_conc**2 * k2
    dS1dt: float = -compartment * reaction1 + compartment * reaction2
    dS2dt: float = 2.0 * compartment * reaction1 - 2.0 * compartment * reaction2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * k1
    reaction2: float = S2_conc**2 * k2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
        "reaction2": reaction2,
    }
