time: float = 0.0
C: float = 1.00000000000000
S1: float = 0.0

# Initial assignments
J0 = 0.100000000000000
S1_stoich = 6.02214179000000
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = 0.100000000000000
    dS1dt: float = J0 * S1_stoich
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    J0: float = 0.100000000000000
    return {
        "J0": J0,
    }
