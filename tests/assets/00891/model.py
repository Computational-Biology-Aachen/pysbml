time: float = 0.0
k3: float = 1.00000000000000
k1: float = 0.000150000000000000
k2: float = 0.0

# Initial assignments
dk1 = -k1 * k3 * time
dk2 = k1 * k3 * time
y0 = [k1, k2]
variable_names = ["k1", "k2"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    k1, k2 = variables
    dk1: float = -k1 * k3 * time
    dk2: float = k1 * k3 * time
    dk1dt: float = dk1
    dk2dt: float = dk2
    return dk1dt, dk2dt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    k1, k2 = variables
    dk1: float = -k1 * k3 * time
    dk2: float = k1 * k3 * time
    return {
        "dk1": dk1,
        "dk2": dk2,
    }
