time: float = 0.0
C: float = 1.00000000000000
S1_stoich: float = 1.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_conc = S1 / C
dS1_stoich = 1.50000000000000
J0 = 0.100000000000000
y0 = [S1, S1_stoich]
variable_names = ["S1", "S1_stoich"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S1_stoich = variables
    dS1_stoich: float = 1.50000000000000
    J0: float = 0.100000000000000
    dS1_stoichdt: float = dS1_stoich
    dS1dt: float = C * J0 * S1_stoich
    return dS1dt, dS1_stoichdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S1_stoich = variables
    S1_conc: float = S1 / C
    dS1_stoich: float = 1.50000000000000
    J0: float = 0.100000000000000
    return {
        "S1_conc": S1_conc,
        "dS1_stoich": dS1_stoich,
        "J0": J0,
    }
