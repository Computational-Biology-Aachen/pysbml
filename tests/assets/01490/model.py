time: float = 0.0

# Initial assignments
a = 1
b = 2
c = 1
d = 1
e = 1
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    a: float = 1
    b: float = 2
    c: float = 1
    d: float = 1
    e: float = 1
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
    }
