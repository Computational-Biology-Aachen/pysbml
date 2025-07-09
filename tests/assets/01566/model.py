import math

time: float = 0.0
C: float = 1.00000000000000
S1_stoich: float = 2.00000000000000
S1: float = 0.0

# Initial assignments
P1 = math.exp(S1_stoich)
P2 = abs(S1_stoich)
P3 = abs(S1_stoich)
P4 = math.acos(1 - S1_stoich)
P5 = math.acos((1 / 4) * S1_stoich)
P6 = math.asin(S1_stoich - 1)
P7 = -math.asin((1 / 4) * S1_stoich)
P8 = math.atan(S1_stoich + 0.8)
P9 = -math.atan(3 * S1_stoich + 1.09)
P10 = math.ceil((1 / 4) * S1_stoich)
P11 = math.ceil(4 * S1_stoich - 0.45)
P12 = math.ceil(-2 * S1_stoich - 0.6)
P13 = math.cos(4 * S1_stoich + 1.1)
P14 = math.cos((1 / 10) * S1_stoich + 0.02)
P15 = 1
P16 = math.exp((1 / 2) * S1_stoich)
P17 = 0.718923733431926 * math.exp((1 / 2) * S1_stoich)
P18 = math.floor(-2 * S1_stoich - 0.6)
P19 = math.floor(4 * S1_stoich + 1.1)
P20 = math.log((1 / 10) * S1_stoich)
P21 = math.log((1 / 2) * S1_stoich)
P22 = math.log((1 / 10) * S1_stoich) / math.log(10)
P23 = math.log((1 / 2) * S1_stoich) / math.log(10)
P24 = 1
P25 = S1_stoich**S1_stoich
P26 = 5.56228172775440
P27 = (S1_stoich**2) ** S1_stoich
P28 = 3.1**S1_stoich
P29 = math.sqrt(2) * math.sqrt(S1_stoich)
P30 = math.sqrt((1 / 5) * S1_stoich + 7)
P31 = math.sin(S1_stoich + 0.1)
P32 = 0
P33 = -math.sin(2 * S1_stoich + 1.9)
P34 = 0
P35 = math.tan((1 / 2) * S1_stoich + 0.11)
P36 = -math.tan(3 * S1_stoich)
P37 = 1 / math.cos((1 / 4) * S1_stoich)
P38 = 1 / math.sin(2.25 * S1_stoich)
P39 = 1 / math.tan((1 / 10) * S1_stoich)
P40 = math.sinh((1 / 10) * S1_stoich + 0.1)
P41 = math.cosh(S1_stoich - 0.3)
P42 = math.acos(1 / (S1_stoich + 0.3))
P43 = math.asin(1 / (S1_stoich - 0.9))
P44 = math.atan(1 / (S1_stoich - 2.1))
P45 = math.asinh(50 * S1_stoich - 1)
P46 = math.acosh((1 / 2) * S1_stoich + 0.34)
P47 = math.atanh(S1_stoich - 2.7)
P48 = math.log(
    math.sqrt(-1 + 4.76190476190476 / S1_stoich)
    * math.sqrt(1 + 4.76190476190476 / S1_stoich)
    + 4.76190476190476 / S1_stoich
)
P49 = math.log(math.sqrt(1 + 40000.0 / S1_stoich**2) + 200.0 / S1_stoich)
S1_conc = S1 / C
J0 = 1
y0 = [S1]
variable_names = ["S1"]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (S1,) = variables
    J0: float = 1
    dS1dt: float = C * J0 * S1_stoich
    return (dS1dt,)


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (S1,) = variables
    P1: float = math.exp(S1_stoich)
    P2: float = abs(S1_stoich)
    P3: float = abs(S1_stoich)
    P4: float = math.acos(1 - S1_stoich)
    P5: float = math.acos((1 / 4) * S1_stoich)
    P6: float = math.asin(S1_stoich - 1)
    P7: float = -math.asin((1 / 4) * S1_stoich)
    P8: float = math.atan(S1_stoich + 0.8)
    P9: float = -math.atan(3 * S1_stoich + 1.09)
    P10: float = math.ceil((1 / 4) * S1_stoich)
    P11: float = math.ceil(4 * S1_stoich - 0.45)
    P12: float = math.ceil(-2 * S1_stoich - 0.6)
    P13: float = math.cos(4 * S1_stoich + 1.1)
    P14: float = math.cos((1 / 10) * S1_stoich + 0.02)
    P15: float = 1
    P16: float = math.exp((1 / 2) * S1_stoich)
    P17: float = 0.718923733431926 * math.exp((1 / 2) * S1_stoich)
    P18: float = math.floor(-2 * S1_stoich - 0.6)
    P19: float = math.floor(4 * S1_stoich + 1.1)
    P20: float = math.log((1 / 10) * S1_stoich)
    P21: float = math.log((1 / 2) * S1_stoich)
    P22: float = math.log((1 / 10) * S1_stoich) / math.log(10)
    P23: float = math.log((1 / 2) * S1_stoich) / math.log(10)
    P24: float = 1
    P25: float = S1_stoich**S1_stoich
    P26: float = 5.56228172775440
    P27: float = (S1_stoich**2) ** S1_stoich
    P28: float = 3.1**S1_stoich
    P29: float = math.sqrt(2) * math.sqrt(S1_stoich)
    P30: float = math.sqrt((1 / 5) * S1_stoich + 7)
    P31: float = math.sin(S1_stoich + 0.1)
    P32: float = 0
    P33: float = -math.sin(2 * S1_stoich + 1.9)
    P34: float = 0
    P35: float = math.tan((1 / 2) * S1_stoich + 0.11)
    P36: float = -math.tan(3 * S1_stoich)
    P37: float = 1 / math.cos((1 / 4) * S1_stoich)
    P38: float = 1 / math.sin(2.25 * S1_stoich)
    P39: float = 1 / math.tan((1 / 10) * S1_stoich)
    P40: float = math.sinh((1 / 10) * S1_stoich + 0.1)
    P41: float = math.cosh(S1_stoich - 0.3)
    P42: float = math.acos(1 / (S1_stoich + 0.3))
    P43: float = math.asin(1 / (S1_stoich - 0.9))
    P44: float = math.atan(1 / (S1_stoich - 2.1))
    P45: float = math.asinh(50 * S1_stoich - 1)
    P46: float = math.acosh((1 / 2) * S1_stoich + 0.34)
    P47: float = math.atanh(S1_stoich - 2.7)
    P48: float = math.log(
        math.sqrt(-1 + 4.76190476190476 / S1_stoich)
        * math.sqrt(1 + 4.76190476190476 / S1_stoich)
        + 4.76190476190476 / S1_stoich
    )
    P49: float = math.log(math.sqrt(1 + 40000.0 / S1_stoich**2) + 200.0 / S1_stoich)
    S1_conc: float = S1 / C
    J0: float = 1
    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "P6": P6,
        "P7": P7,
        "P8": P8,
        "P9": P9,
        "P10": P10,
        "P11": P11,
        "P12": P12,
        "P13": P13,
        "P14": P14,
        "P15": P15,
        "P16": P16,
        "P17": P17,
        "P18": P18,
        "P19": P19,
        "P20": P20,
        "P21": P21,
        "P22": P22,
        "P23": P23,
        "P24": P24,
        "P25": P25,
        "P26": P26,
        "P27": P27,
        "P28": P28,
        "P29": P29,
        "P30": P30,
        "P31": P31,
        "P32": P32,
        "P33": P33,
        "P34": P34,
        "P35": P35,
        "P36": P36,
        "P37": P37,
        "P38": P38,
        "P39": P39,
        "P40": P40,
        "P41": P41,
        "P42": P42,
        "P43": P43,
        "P44": P44,
        "P45": P45,
        "P46": P46,
        "P47": P47,
        "P48": P48,
        "P49": P49,
        "S1_conc": S1_conc,
        "J0": J0,
    }
