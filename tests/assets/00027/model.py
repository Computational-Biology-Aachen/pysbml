time: float = 0.0
reaction1_k: float = 100.000000000000
S1: float = 0.0150000000000000
S2: float = 0.0

# Initial assignments
compartment = 0.534000000000000
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = S1_conc * reaction1_k
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * reaction1_k
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * reaction1_k
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
