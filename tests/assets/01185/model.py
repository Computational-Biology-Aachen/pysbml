time: float = 0.0
y: float = 3.00000000000000
C: float = 1.00000000000000
S1: float = 1.00000000000000

# Initial assignments
S1_conc = S1 / C
k1 = y
__J0 = k1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    __J0: float = k1
    dS1dt: float = C * __J0
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_conc: float = S1 / C
    __J0: float = k1
    return {
        "S1_conc": S1_conc,
        "__J0": __J0,
    }
