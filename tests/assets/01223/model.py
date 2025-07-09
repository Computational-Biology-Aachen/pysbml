time: float = 0.0
S1: float = 1.20000000000000
S1: float = 1.20000000000000

# Initial assignments
c = time + 2
S1_conc = S1 / c
dS1 = 1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    c: float = time + 2
    dS1: float = 1
    dS1dt: float = dS1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    c: float = time + 2
    S1_conc: float = S1 / c
    dS1: float = 1
    return {
        "c": c,
        "S1_conc": S1_conc,
        "dS1": dS1,
    }
