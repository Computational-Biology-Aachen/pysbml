import math

time: float = 0.0
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
dP2 = 1
dP3 = 1
dP4 = math.pi
dP5 = 1.04719755119660
dP6 = (1 / 2) * math.pi
dP7 = -0.523598775598299
dP8 = 1.22777238637419
dP9 = -1.43067687253053
dP10 = 1
dP11 = 4
dP12 = -4
dP13 = -0.947721602131112
dP14 = 0.975897449330606
dP15 = 1
dP16 = math.e
dP17 = 2.15976625378492
dP18 = -5
dP19 = 9
dP20 = -1.60943791243410
dP21 = 0
dP22 = -1.6094379124341 / math.log(10)
dP23 = 0
dP24 = 1
dP25 = 4
dP26 = 5.56228172775440
dP27 = 16
dP28 = 9.61000000000000
dP29 = 2
dP30 = 2.72029410174709
dP31 = 0.863209366648874
dP32 = 0
dP33 = 0.373876664830236
dP34 = 0
dP35 = 2.01433821447683
dP36 = -math.tan(6)
dP37 = 0
dP38 = 0.804062391404892
dP39 = -math.tanh(6)
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
    dP2: float = 1
    dP3: float = 1
    dP4: float = math.pi
    dP5: float = 1.04719755119660
    dP6: float = (1 / 2) * math.pi
    dP7: float = -0.523598775598299
    dP8: float = 1.22777238637419
    dP9: float = -1.43067687253053
    dP10: float = 1
    dP11: float = 4
    dP12: float = -4
    dP13: float = -0.947721602131112
    dP14: float = 0.975897449330606
    dP15: float = 1
    dP16: float = math.e
    dP17: float = 2.15976625378492
    dP18: float = -5
    dP19: float = 9
    dP20: float = -1.60943791243410
    dP21: float = 0
    dP22: float = -1.6094379124341 / math.log(10)
    dP23: float = 0
    dP24: float = 1
    dP25: float = 4
    dP26: float = 5.56228172775440
    dP27: float = 16
    dP28: float = 9.61000000000000
    dP29: float = 2
    dP30: float = 2.72029410174709
    dP31: float = 0.863209366648874
    dP32: float = 0
    dP33: float = 0.373876664830236
    dP34: float = 0
    dP35: float = 2.01433821447683
    dP36: float = -math.tan(6)
    dP37: float = 0
    dP38: float = 0.804062391404892
    dP39: float = -math.tanh(6)
    dP2dt: float = dP2
    dP3dt: float = dP3
    dP4dt: float = dP4
    dP5dt: float = dP5
    dP6dt: float = dP6
    dP7dt: float = dP7
    dP8dt: float = dP8
    dP9dt: float = dP9
    dP10dt: float = dP10
    dP11dt: float = dP11
    dP12dt: float = dP12
    dP13dt: float = dP13
    dP14dt: float = dP14
    dP15dt: float = dP15
    dP16dt: float = dP16
    dP17dt: float = dP17
    dP18dt: float = dP18
    dP19dt: float = dP19
    dP20dt: float = dP20
    dP21dt: float = dP21
    dP22dt: float = dP22
    dP23dt: float = dP23
    dP24dt: float = dP24
    dP25dt: float = dP25
    dP26dt: float = dP26
    dP27dt: float = dP27
    dP28dt: float = dP28
    dP29dt: float = dP29
    dP30dt: float = dP30
    dP31dt: float = dP31
    dP32dt: float = dP32
    dP33dt: float = dP33
    dP34dt: float = dP34
    dP35dt: float = dP35
    dP36dt: float = dP36
    dP37dt: float = dP37
    dP38dt: float = dP38
    dP39dt: float = dP39
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
    dP2: float = 1
    dP3: float = 1
    dP4: float = math.pi
    dP5: float = 1.04719755119660
    dP6: float = (1 / 2) * math.pi
    dP7: float = -0.523598775598299
    dP8: float = 1.22777238637419
    dP9: float = -1.43067687253053
    dP10: float = 1
    dP11: float = 4
    dP12: float = -4
    dP13: float = -0.947721602131112
    dP14: float = 0.975897449330606
    dP15: float = 1
    dP16: float = math.e
    dP17: float = 2.15976625378492
    dP18: float = -5
    dP19: float = 9
    dP20: float = -1.60943791243410
    dP21: float = 0
    dP22: float = -1.6094379124341 / math.log(10)
    dP23: float = 0
    dP24: float = 1
    dP25: float = 4
    dP26: float = 5.56228172775440
    dP27: float = 16
    dP28: float = 9.61000000000000
    dP29: float = 2
    dP30: float = 2.72029410174709
    dP31: float = 0.863209366648874
    dP32: float = 0
    dP33: float = 0.373876664830236
    dP34: float = 0
    dP35: float = 2.01433821447683
    dP36: float = -math.tan(6)
    dP37: float = 0
    dP38: float = 0.804062391404892
    dP39: float = -math.tanh(6)
    return {
        "dP2": dP2,
        "dP3": dP3,
        "dP4": dP4,
        "dP5": dP5,
        "dP6": dP6,
        "dP7": dP7,
        "dP8": dP8,
        "dP9": dP9,
        "dP10": dP10,
        "dP11": dP11,
        "dP12": dP12,
        "dP13": dP13,
        "dP14": dP14,
        "dP15": dP15,
        "dP16": dP16,
        "dP17": dP17,
        "dP18": dP18,
        "dP19": dP19,
        "dP20": dP20,
        "dP21": dP21,
        "dP22": dP22,
        "dP23": dP23,
        "dP24": dP24,
        "dP25": dP25,
        "dP26": dP26,
        "dP27": dP27,
        "dP28": dP28,
        "dP29": dP29,
        "dP30": dP30,
        "dP31": dP31,
        "dP32": dP32,
        "dP33": dP33,
        "dP34": dP34,
        "dP35": dP35,
        "dP36": dP36,
        "dP37": dP37,
        "dP38": dP38,
        "dP39": dP39,
    }
