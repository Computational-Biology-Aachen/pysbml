time: float = 0.0
C: float = 1.00000000000000
S1: float = 2.00000000000000
S2: float = 3.00000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
J0 = 0.100000000000000
y0 = [S2]
variable_names = ["S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S2,) = variables
    J0: float = 0.100000000000000
    dS2dt: float = 2.0 * C * J0
    return (dS2dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S2,) = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    J0: float = 0.100000000000000
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "J0": J0,
    }
