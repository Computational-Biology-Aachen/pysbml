time: float = 0.0
c: float = 1.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_conc = S1 / c
J0 = 2
J1 = 1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J1: float = 1
    dS1dt: float = J1 * c
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_conc: float = S1 / c
    J0: float = 2
    J1: float = 1
    return {
        "S1_conc": S1_conc,
        "J0": J0,
        "J1": J1,
    }
