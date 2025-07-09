time: float = 0.0
C: float = 1.00000000000000
J0_avogadro: float = 0.100000000000000
S1: float = 1.00000000000000

# Initial assignments
J0 = J0_avogadro
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = J0_avogadro
    dS1dt: float = J0
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = J0_avogadro
    return {
        "J0": J0,
    }
