time: float = 0.0
c: float = 1.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_conc = S1 / c
J0 = 2
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_conc: float = S1 / c
    J0: float = 2
    return {
        "S1_conc": S1_conc,
        "J0": J0,
    }
