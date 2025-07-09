time: float = 0.0
C: float = 1.00000000000000
S2_create: float = 1.00000000000000
S1: float = 30.0000000000000
S2: float = 0.0

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
dS2_create = 1
J0 = 1
S1_degrade = 3
y0 = [S1, S2, S2_create]
variable_names = ["S1", "S2", "S2_create"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, S2, S2_create = variables
    dS2_create: float = 1
    J0: float = 1
    dS2_createdt: float = dS2_create
    dS1dt: float = -C * J0 * S1_degrade
    dS2dt: float = C * J0 * S2_create
    return dS1dt, dS2dt, dS2_createdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, S2, S2_create = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    dS2_create: float = 1
    J0: float = 1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "dS2_create": dS2_create,
        "J0": J0,
    }
