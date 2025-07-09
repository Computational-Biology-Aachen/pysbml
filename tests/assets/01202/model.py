time: float = 0.0
x: float = 0.0
y: float = 2.00000000000000

# Initial assignments
z = (2) if (x <= 0.49) else ((1) if (y > 0.49) else (0))
dx = 1
dy = -2
y0 = [x, y]
variable_names = ["x", "y"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    x, y = variables
    dx: float = 1
    dy: float = -2
    dxdt: float = dx
    dydt: float = dy
    return dxdt, dydt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    x, y = variables
    z: float = (2) if (x <= 0.49) else ((1) if (y > 0.49) else (0))
    dx: float = 1
    dy: float = -2
    return {
        "z": z,
        "dx": dx,
        "dy": dy,
    }
