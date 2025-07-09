time: float = 0.0
C: float = 1.00000000000000
S1_conc: float = 3.00000000000000
S2: float = 3.00000000000000

# Initial assignments
S1_stoich = 2
S1 = C * S1_conc
J0 = 0.1 * S1
y0 = [S2]
variable_names = ["S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S2,) = variables
    S1_stoich: float = 2
    S1: float = C * S1_conc
    J0: float = 0.1 * S1
    dS2dt: float = J0
    dS1dt: float = -J0 * S1_stoich
    return (dS2dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S2,) = variables
    S1_stoich: float = 2
    S1: float = C * S1_conc
    J0: float = 0.1 * S1
    return {
        "S1_stoich": S1_stoich,
        "S1": S1,
        "J0": J0,
    }
