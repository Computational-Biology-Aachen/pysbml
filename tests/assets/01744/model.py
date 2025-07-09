time: float = 0.0
C: float = 1.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_stoich = (1 / 10) * time
S1_conc = S1 / C
J0 = 0.100000000000000
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    S1_stoich: float = (1 / 10) * time
    J0: float = 0.100000000000000
    dS1dt: float = C * J0 * S1_stoich
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_stoich: float = (1 / 10) * time
    S1_conc: float = S1 / C
    J0: float = 0.100000000000000
    return {
        "S1_stoich": S1_stoich,
        "S1_conc": S1_conc,
        "J0": J0,
    }
