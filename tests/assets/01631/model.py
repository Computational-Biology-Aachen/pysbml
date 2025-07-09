time: float = 0.0
C: float = 1.00000000000000
S1_create: float = 1.00000000000000
S1: float = 2.00000000000000

# Initial assignments
S1_conc = S1 / C
dS1_create = 1
J0 = 1
J1 = 1
S1_degrade = 3
y0 = [S1, S1_create]
variable_names = ["S1", "S1_create"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S1_create = variables
    dS1_create: float = 1
    J0: float = 1
    J1: float = 1
    dS1_createdt: float = dS1_create
    dS1dt: float = -C * J0 * S1_degrade + C * J1 * S1_create
    return dS1dt, dS1_createdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S1_create = variables
    S1_conc: float = S1 / C
    dS1_create: float = 1
    J0: float = 1
    J1: float = 1
    return {
        "S1_conc": S1_conc,
        "dS1_create": dS1_create,
        "J0": J0,
        "J1": J1,
    }
