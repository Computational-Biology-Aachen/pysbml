time: float = 0.0
compartment: float = 1.00000000000000

# Initial assignments
S1 = 7
S1_conc = S1 / compartment
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1: float = 7
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1: float = 7
    S1_conc: float = S1 / compartment
    return {
        "S1": S1,
        "S1_conc": S1_conc,
    }
