time: float = 0.0
avogadro: float = 0.100000000000000
C: float = 1.00000000000000
S1: float = 1.00000000000000

# Initial assignments
J0 = avogadro
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = avogadro
    dS1dt: float = J0
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = avogadro
    return {
        "J0": J0,
    }
