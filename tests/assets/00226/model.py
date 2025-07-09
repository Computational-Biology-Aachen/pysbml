time: float = 0.0
k1: float = 1.50000000000000
compartment: float = 1.00000000000000
S1: float = 1.50000000000000e-6
S2: float = 1.00000000000000e-6

# Initial assignments
S1_conc = S1 / compartment
S2_conc = S2 / compartment
reaction1 = S1_conc * k1
y0 = [S2]
variable_names = ["S2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S2,) = variables
    S1_conc: float = S1 / compartment
    reaction1: float = S1_conc * k1
    dS2dt: float = compartment * reaction1
    return (dS2dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S2,) = variables
    S1_conc: float = S1 / compartment
    S2_conc: float = S2 / compartment
    reaction1: float = S1_conc * k1
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
