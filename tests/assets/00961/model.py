time: float = 0.0

# Initial assignments
P1 = 6.02214179000000e23
P2 = 6.02214179000000e23
P3 = 6.02214179000000e23
P4 = 6.02214179000000e23
P5 = 4.17899999992689e18
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    P1: float = 6.02214179000000e23
    P2: float = 6.02214179000000e23
    P3: float = 6.02214179000000e23
    P4: float = 6.02214179000000e23
    P5: float = 4.17899999992689e18
    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
    }
