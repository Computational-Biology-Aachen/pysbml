import math

time: float = 0.0
p1: float = 10.5000000000000
p2: float = 10.0000000000000
compartment: float = 1.00000000000000
S1: float = 1.00000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = math.ceil(S1_conc * p1) / p2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    reaction1: float = math.ceil(S1_conc * p1) / p2
    dS1dt: float = -compartment * reaction1
    dS2dt: float = compartment * reaction1
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = math.ceil(S1_conc * p1) / p2
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
