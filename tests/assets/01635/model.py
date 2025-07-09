time: float = 0.0
C: float = 1.00000000000000
S1: float = 2.00000000000000
S1_stoich: float = 1.00000000000000

# Initial assignments
S1_conc = S1 / C
dS1_stoich = 1
J0 = 1
y0 = [S1_stoich]
variable_names = ["S1_stoich"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1_stoich,) = variables
    dS1_stoich: float = 1
    J0: float = 1
    dS1_stoichdt: float = dS1_stoich
    dS1dt: float = -J0 * S1_stoich
    return (dS1_stoichdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1_stoich,) = variables
    S1_conc: float = S1 / C
    dS1_stoich: float = 1
    J0: float = 1
    return {
        "S1_conc": S1_conc,
        "dS1_stoich": dS1_stoich,
        "J0": J0,
    }
