time: float = 0.0
x: float = 0.0

# Initial assignments
z = (1) if (x >= 0.49) else (0)
dx = 1
y0 = [x]
variable_names = ["x"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (x,) = variables
    dx: float = 1
    dxdt: float = dx
    return (dxdt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (x,) = variables
    z: float = (1) if (x >= 0.49) else (0)
    dx: float = 1
    return {
        "z": z,
        "dx": dx,
    }
