time: float = 0.0
C: float = 2.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_conc = S1 / C
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1_conc: float = S1 / C
    return {
        "S1_conc": S1_conc,
    }
