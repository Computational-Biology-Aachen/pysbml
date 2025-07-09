time: float = 0.0
avogadro: float = 1.00000000000000e24
C: float = 1.00000000000000
S1: float = 1.00000000000000

# Initial assignments
J0 = 6.02214179e23 / avogadro
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = 6.02214179e23 / avogadro
    dS1dt: float = J0
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = 6.02214179e23 / avogadro
    return {
        "J0": J0,
    }
