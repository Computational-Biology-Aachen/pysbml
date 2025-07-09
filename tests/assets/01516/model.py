import math

time: float = 0.0
C: float = 1.00000000000000
P2: float = 0.0
P3: float = 0.0
P4: float = 0.0
P5: float = 0.0
P6: float = 0.0
P7: float = 0.0
P8: float = 0.0
P9: float = 0.0
P10: float = 0.0
P11: float = 0.0
P12: float = 0.0
P13: float = 0.0
P14: float = 0.0
P15: float = 0.0
P16: float = 0.0
P17: float = 0.0
P18: float = 0.0
P19: float = 0.0
P20: float = 0.0
P21: float = 0.0
P22: float = 0.0
P23: float = 0.0
P24: float = 0.0
P25: float = 0.0
P26: float = 0.0
P27: float = 0.0
P28: float = 0.0
P29: float = 0.0
P30: float = 0.0
P31: float = 0.0
P32: float = 0.0
P33: float = 0.0
P34: float = 0.0
P35: float = 0.0
P36: float = 0.0
P37: float = 0.0
P38: float = 0.0
P39: float = 0.0

# Initial assignments
J0 = 1
P2_sr = 1
P3_sr = 1
P4_sr = math.pi
P5_sr = 1.04719755119660
P6_sr = (1 / 2) * math.pi
P7_sr = -0.523598775598299
P8_sr = 1.22777238637419
P9_sr = -1.43067687253053
P10_sr = 1
P11_sr = 4
P12_sr = -4
P13_sr = -0.947721602131112
P14_sr = 0.975897449330606
P15_sr = 1
P16_sr = math.e
P17_sr = 2.15976625378492
P18_sr = -5
P19_sr = 9
P20_sr = -1.60943791243410
P21_sr = 0
P22_sr = -1.6094379124341 / math.log(10)
P23_sr = 0
P24_sr = 1
P25_sr = 4
P26_sr = 5.56228172775440
P27_sr = 16
P28_sr = 9.61000000000000
P29_sr = 2
P30_sr = 2.72029410174709
P31_sr = 0.863209366648874
P32_sr = 0
P33_sr = 0.373876664830236
P34_sr = 0
P35_sr = 2.01433821447683
P36_sr = -math.tan(6)
P37_sr = 0
P38_sr = 0.804062391404892
P39_sr = -math.tanh(6)
y0 = [
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
    P16,
    P17,
    P18,
    P19,
    P2,
    P20,
    P21,
    P22,
    P23,
    P24,
    P25,
    P26,
    P27,
    P28,
    P29,
    P3,
    P30,
    P31,
    P32,
    P33,
    P34,
    P35,
    P36,
    P37,
    P38,
    P39,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
]
variable_names = [
    "P10",
    "P11",
    "P12",
    "P13",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P19",
    "P2",
    "P20",
    "P21",
    "P22",
    "P23",
    "P24",
    "P25",
    "P26",
    "P27",
    "P28",
    "P29",
    "P3",
    "P30",
    "P31",
    "P32",
    "P33",
    "P34",
    "P35",
    "P36",
    "P37",
    "P38",
    "P39",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (
        P10,
        P11,
        P12,
        P13,
        P14,
        P15,
        P16,
        P17,
        P18,
        P19,
        P2,
        P20,
        P21,
        P22,
        P23,
        P24,
        P25,
        P26,
        P27,
        P28,
        P29,
        P3,
        P30,
        P31,
        P32,
        P33,
        P34,
        P35,
        P36,
        P37,
        P38,
        P39,
        P4,
        P5,
        P6,
        P7,
        P8,
        P9,
    ) = variables
    J0: float = 1
    dP2dt: float = J0 * P2_sr
    dP3dt: float = J0 * P3_sr
    dP4dt: float = J0 * P4_sr
    dP5dt: float = J0 * P5_sr
    dP6dt: float = J0 * P6_sr
    dP7dt: float = J0 * P7_sr
    dP8dt: float = J0 * P8_sr
    dP9dt: float = J0 * P9_sr
    dP10dt: float = J0 * P10_sr
    dP11dt: float = J0 * P11_sr
    dP12dt: float = J0 * P12_sr
    dP13dt: float = J0 * P13_sr
    dP14dt: float = J0 * P14_sr
    dP15dt: float = J0 * P15_sr
    dP16dt: float = J0 * P16_sr
    dP17dt: float = J0 * P17_sr
    dP18dt: float = J0 * P18_sr
    dP19dt: float = J0 * P19_sr
    dP20dt: float = J0 * P20_sr
    dP21dt: float = J0 * P21_sr
    dP22dt: float = J0 * P22_sr
    dP23dt: float = J0 * P23_sr
    dP24dt: float = J0 * P24_sr
    dP25dt: float = J0 * P25_sr
    dP26dt: float = J0 * P26_sr
    dP27dt: float = J0 * P27_sr
    dP28dt: float = J0 * P28_sr
    dP29dt: float = J0 * P29_sr
    dP30dt: float = J0 * P30_sr
    dP31dt: float = J0 * P31_sr
    dP32dt: float = J0 * P32_sr
    dP33dt: float = J0 * P33_sr
    dP34dt: float = J0 * P34_sr
    dP35dt: float = J0 * P35_sr
    dP36dt: float = J0 * P36_sr
    dP37dt: float = J0 * P37_sr
    dP38dt: float = J0 * P38_sr
    dP39dt: float = J0 * P39_sr
    return (
        dP10dt,
        dP11dt,
        dP12dt,
        dP13dt,
        dP14dt,
        dP15dt,
        dP16dt,
        dP17dt,
        dP18dt,
        dP19dt,
        dP2dt,
        dP20dt,
        dP21dt,
        dP22dt,
        dP23dt,
        dP24dt,
        dP25dt,
        dP26dt,
        dP27dt,
        dP28dt,
        dP29dt,
        dP3dt,
        dP30dt,
        dP31dt,
        dP32dt,
        dP33dt,
        dP34dt,
        dP35dt,
        dP36dt,
        dP37dt,
        dP38dt,
        dP39dt,
        dP4dt,
        dP5dt,
        dP6dt,
        dP7dt,
        dP8dt,
        dP9dt,
    )


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (
        P10,
        P11,
        P12,
        P13,
        P14,
        P15,
        P16,
        P17,
        P18,
        P19,
        P2,
        P20,
        P21,
        P22,
        P23,
        P24,
        P25,
        P26,
        P27,
        P28,
        P29,
        P3,
        P30,
        P31,
        P32,
        P33,
        P34,
        P35,
        P36,
        P37,
        P38,
        P39,
        P4,
        P5,
        P6,
        P7,
        P8,
        P9,
    ) = variables
    J0: float = 1
    return {
        "J0": J0,
    }
