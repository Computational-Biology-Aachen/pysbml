time: float = 0.0
C1: float = 0.500000000000000
S1: float = 3.00000000000000

# Initial assignments
x = S1 / C1
S1_conc = S1 / C1
dC1 = 0.200000000000000
dS1 = S1 * dC1 - 0.2 * S1
y0 = [C1, S1]
variable_names = ["C1", "S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    C1, S1 = variables
    dC1: float = 0.200000000000000
    dS1: float = S1 * dC1 - 0.2 * S1
    dS1dt: float = dS1
    dC1dt: float = dC1
    return dC1dt, dS1dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    C1, S1 = variables
    x: float = S1 / C1
    S1_conc: float = S1 / C1
    dC1: float = 0.200000000000000
    dS1: float = S1 * dC1 - 0.2 * S1
    return {
        "x": x,
        "S1_conc": S1_conc,
        "dC1": dC1,
        "dS1": dS1,
    }
