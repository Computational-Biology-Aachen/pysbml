time: float = 0.0
y: float = 2.00000000000000
x: float = 0.0

# Initial assignments
z = ((2) if (y > 1.49) else (1)) if (x <= 0.49) else (0)
dy = -2
dx = 1
y0 = [x, y]
variable_names = ["x", "y"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    x, y = variables
    dy: float = -2
    dx: float = 1
    dydt: float = dy
    dxdt: float = dx
    return dxdt, dydt


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    x, y = variables
    z: float = ((2) if (y > 1.49) else (1)) if (x <= 0.49) else (0)
    dy: float = -2
    dx: float = 1
    return {
        "z": z,
        "dy": dy,
        "dx": dx,
    }
