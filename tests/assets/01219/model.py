time: float = 0.0
c: float = 2.00000000000000
S1: float = 1.20000000000000
S1: float = 1.20000000000000

# Initial assignments
S1_conc = S1 / c
dS1 = 1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    dS1: float = 1
    dS1dt: float = dS1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_conc: float = S1 / c
    dS1: float = 1
    return {
        "S1_conc": S1_conc,
        "dS1": dS1,
    }
