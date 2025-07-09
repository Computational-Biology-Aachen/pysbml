time: float = 0.0
k1: float = 1.00000000000000
compartment: float = 1.00000000000000
S1: float = 0.0150000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
dS1 = -S1 * k1
dS2 = S1 * k1
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    dS1: float = -S1 * k1
    dS2: float = S1 * k1
    dS1dt: float = compartment * dS1
    dS2dt: float = compartment * dS2
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    dS1: float = -S1 * k1
    dS2: float = S1 * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "dS1": dS1,
        "dS2": dS2,
    }
