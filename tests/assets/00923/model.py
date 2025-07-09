time: float = 0.0
k2: float = 0.300000000000000

# Initial assignments
k1 = 4 * k2
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    k1: float = 4 * k2
    return {
        "k1": k1,
    }
