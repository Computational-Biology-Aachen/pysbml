time: float = 0.0
kf: float = 0.900000000000000
kr: float = 0.0750000000000000
C: float = 1.00000000000000
S2: float = 0.0
S1: float = 1.00000000000000

# Initial assignments
S1_conc = S1 / C
S2_conc = S2 / C
reaction1 = S1_conc * kf - S2_conc * kr
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    reaction1: float = S1_conc * kf - S2_conc * kr
    dS1dt: float = -reaction1
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    S1_conc: float = S1 / C
    S2_conc: float = S2 / C
    reaction1: float = S1_conc * kf - S2_conc * kr
    return {
        "S1_conc": S1_conc,
        "S2_conc": S2_conc,
        "reaction1": reaction1,
    }
