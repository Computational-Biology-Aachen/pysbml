time: float = 0.0

# Initial assignments
a = 2
b = 4
c = 1
d = 1
e = 0
y0 = []
variable_names = []


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    return


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    a: float = 2
    b: float = 4
    c: float = 1
    d: float = 1
    e: float = 0
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
    }
