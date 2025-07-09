time: float = 0.0
k1: float = 1.00000000000000
C: float = 1.00000000000000
S2: float = 0.0
S1: float = 0.000150000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
reaction1 = S1_conc * k1 * time
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    S1_conc: float = S1 / C
    reaction1: float = S1_conc * k1 * time
    dS1dt: float = -C * reaction1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    reaction1: float = S1_conc * k1 * time
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
