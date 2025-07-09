time: float = 0.0
S3: float = 4.00000000000000
comp: float = 5.00000000000000
S1: float = 1.00000000000000

# Initial assignments
S1_conc = S1 / comp
S3_conc = S3 / comp
dcomp = 1
__J0 = (1 / 10) * S3_conc
y0 = [S1, comp]
variable_names = ["S1", "comp"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    S1, comp = variables
    S3_conc: float = S3 / comp
    dcomp: float = 1
    __J0: float = (1 / 10) * S3_conc
    dcompdt: float = dcomp
    dS1dt: float = -__J0
    return dS1dt, dcompdt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    S1, comp = variables
    S1_conc: float = S1 / comp
    S3_conc: float = S3 / comp
    dcomp: float = 1
    __J0: float = (1 / 10) * S3_conc
    return {
        "S1_conc": S1_conc,
        "S3_conc": S3_conc,
        "dcomp": dcomp,
        "__J0": __J0,
    }
