time: float = 0.0
C: float = 1.00000000000000
S1_stoich: float = 2.00000000000000
S1: float = 2.00000000000000
S2: float = 3.00000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
J0 = 0.0100000000000000
k0 = S1_stoich + 2
y0 = [S1, S2]
variable_names = ["S1", "S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2 = variables
    J0: float = 0.0100000000000000
    dS2dt: float = C * J0
    dS1dt: float = -C * J0 * S1_stoich
    return dS1dt, dS2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2 = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    J0: float = 0.0100000000000000
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "J0": J0,
    }
