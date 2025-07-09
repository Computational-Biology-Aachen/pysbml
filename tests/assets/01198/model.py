time: float = 0.0
C1: float = 0.500000000000000
S1_conc: float = 0.0

# Initial assignments
x = S1_conc
S1 = C1 * S1_conc
dS1 = 0.400000000000000
dC1 = 0.200000000000000
y0 = [C1, S1_conc]
variable_names = ["C1", "S1_conc"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C1, S1_conc = variables
    dS1: float = 0.400000000000000
    dC1: float = 0.200000000000000
    dS1_concdt: float = dS1
    dC1dt: float = dC1
    return dC1dt, dS1_concdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    C1, S1_conc = variables
    x: float = S1_conc
    S1: float = C1 * S1_conc
    dS1: float = 0.400000000000000
    dC1: float = 0.200000000000000
    return {
        "x": x,
        "S1": S1,
        "dS1": dS1,
        "dC1": dC1,
    }
