import math

time: float = 0.0
C: float = 1.00000000000000
S0: float = 0.0
S1: float = 0.0
S2: float = 0.0
S3: float = 0.0
S4: float = 0.0
S5: float = 0.0
S6: float = 0.0
S7: float = 0.0
S8: float = 0.0
S9: float = 0.0
S10: float = 0.0
S11: float = 0.0
S12: float = 0.0
S13: float = 0.0
S14: float = 0.0
S15: float = 0.0
S16: float = 0.0
S17: float = 0.0
S18: float = 0.0
S19: float = 0.0
S20: float = 0.0
S21: float = 0.0
S22: float = 0.0
S23: float = 0.0
S24: float = 0.0
S25: float = 0.0
S26: float = 0.0
S27: float = 0.0
S28: float = 0.0
S29: float = 0.0
S30: float = 0.0
S31: float = 0.0
S32: float = 0.0
S33: float = 0.0
S34: float = 0.0
S35: float = 0.0
S36: float = 0.0
S37: float = 0.0
S38: float = 0.0
S39: float = 0.0
S40: float = 0.0
S41: float = 0.0
S42: float = 0.0
S43: float = 0.0
S44: float = 0.0
S45: float = 0.0
S46: float = 0.0
S47: float = 0.0
S48: float = 0.0
S49: float = 0.0
S50: float = 0.0
S51: float = 0.0

# Initial assignments
J0 = math.e
J1 = math.exp(math.e)
J2 = 1
J3 = 1
J4 = math.pi
J5 = 1.04719755119660
J6 = (1 / 2) * math.pi
J7 = -0.523598775598299
J8 = 1.22777238637419
J9 = -1.43067687253053
J10 = 1
J11 = 4
J12 = -4
J13 = -0.947721602131112
J14 = 0.975897449330606
J15 = 1
J16 = math.e
J17 = 2.15976625378492
J18 = -5
J19 = 9
J20 = -1.60943791243410
J21 = 0
J22 = -1.6094379124341 / math.log(10)
J23 = 0
J24 = 1
J25 = 4
J26 = 5.56228172775440
J27 = 16
J28 = 9.61000000000000
J29 = 2
J30 = 2.72029410174709
J31 = 0.863209366648874
J32 = 0
J33 = 0.373876664830236
J34 = 0
J35 = 2.01433821447683
J36 = -math.tan(6)
J37 = 1.13949392732455
J38 = -1.02298638367130
J39 = 4.93315487558689
J40 = 0.304520293447143
J41 = 2.82831545788997
J42 = 1.12099946637054
J43 = 1.14109666064347
J44 = -1.47112767430373
J45 = math.asinh(99)
J46 = 0.802881971289134
J47 = -0.867300527694053
J48 = 1.51330668842682
J49 = 5.29834236561059
J50 = 1
J51 = 0
y0 = [
    S0,
    S1,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
    S17,
    S18,
    S19,
    S2,
    S20,
    S21,
    S22,
    S23,
    S24,
    S25,
    S26,
    S27,
    S28,
    S29,
    S3,
    S30,
    S31,
    S32,
    S33,
    S34,
    S35,
    S36,
    S37,
    S38,
    S39,
    S4,
    S40,
    S41,
    S42,
    S43,
    S44,
    S45,
    S46,
    S47,
    S48,
    S49,
    S5,
    S50,
    S51,
    S6,
    S7,
    S8,
    S9,
]
variable_names = [
    "S0",
    "S1",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S2",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",
    "S26",
    "S27",
    "S28",
    "S29",
    "S3",
    "S30",
    "S31",
    "S32",
    "S33",
    "S34",
    "S35",
    "S36",
    "S37",
    "S38",
    "S39",
    "S4",
    "S40",
    "S41",
    "S42",
    "S43",
    "S44",
    "S45",
    "S46",
    "S47",
    "S48",
    "S49",
    "S5",
    "S50",
    "S51",
    "S6",
    "S7",
    "S8",
    "S9",
]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (
        S0,
        S1,
        S10,
        S11,
        S12,
        S13,
        S14,
        S15,
        S16,
        S17,
        S18,
        S19,
        S2,
        S20,
        S21,
        S22,
        S23,
        S24,
        S25,
        S26,
        S27,
        S28,
        S29,
        S3,
        S30,
        S31,
        S32,
        S33,
        S34,
        S35,
        S36,
        S37,
        S38,
        S39,
        S4,
        S40,
        S41,
        S42,
        S43,
        S44,
        S45,
        S46,
        S47,
        S48,
        S49,
        S5,
        S50,
        S51,
        S6,
        S7,
        S8,
        S9,
    ) = variables
    J0: float = math.e
    J1: float = math.exp(math.e)
    J2: float = 1
    J3: float = 1
    J4: float = math.pi
    J5: float = 1.04719755119660
    J6: float = (1 / 2) * math.pi
    J7: float = -0.523598775598299
    J8: float = 1.22777238637419
    J9: float = -1.43067687253053
    J10: float = 1
    J11: float = 4
    J12: float = -4
    J13: float = -0.947721602131112
    J14: float = 0.975897449330606
    J15: float = 1
    J16: float = math.e
    J17: float = 2.15976625378492
    J18: float = -5
    J19: float = 9
    J20: float = -1.60943791243410
    J21: float = 0
    J22: float = -1.6094379124341 / math.log(10)
    J23: float = 0
    J24: float = 1
    J25: float = 4
    J26: float = 5.56228172775440
    J27: float = 16
    J28: float = 9.61000000000000
    J29: float = 2
    J30: float = 2.72029410174709
    J31: float = 0.863209366648874
    J32: float = 0
    J33: float = 0.373876664830236
    J34: float = 0
    J35: float = 2.01433821447683
    J36: float = -math.tan(6)
    J37: float = 1.13949392732455
    J38: float = -1.02298638367130
    J39: float = 4.93315487558689
    J40: float = 0.304520293447143
    J41: float = 2.82831545788997
    J42: float = 1.12099946637054
    J43: float = 1.14109666064347
    J44: float = -1.47112767430373
    J45: float = math.asinh(99)
    J46: float = 0.802881971289134
    J47: float = -0.867300527694053
    J48: float = 1.51330668842682
    J49: float = 5.29834236561059
    J50: float = 1
    J51: float = 0
    dS0dt: float = J0
    dS1dt: float = J1
    dS2dt: float = J2
    dS3dt: float = J3
    dS4dt: float = J4
    dS5dt: float = J5
    dS6dt: float = J6
    dS7dt: float = J7
    dS8dt: float = J8
    dS9dt: float = J9
    dS10dt: float = J10
    dS11dt: float = J11
    dS12dt: float = J12
    dS13dt: float = J13
    dS14dt: float = J14
    dS15dt: float = J15
    dS16dt: float = J16
    dS17dt: float = J17
    dS18dt: float = J18
    dS19dt: float = J19
    dS20dt: float = J20
    dS21dt: float = J21
    dS22dt: float = J22
    dS23dt: float = J23
    dS24dt: float = J24
    dS25dt: float = J25
    dS26dt: float = J26
    dS27dt: float = J27
    dS28dt: float = J28
    dS29dt: float = J29
    dS30dt: float = J30
    dS31dt: float = J31
    dS32dt: float = J32
    dS33dt: float = J33
    dS34dt: float = J34
    dS35dt: float = J35
    dS36dt: float = J36
    dS37dt: float = J37
    dS38dt: float = J38
    dS39dt: float = J39
    dS40dt: float = J40
    dS41dt: float = J41
    dS42dt: float = J42
    dS43dt: float = J43
    dS44dt: float = J44
    dS45dt: float = J45
    dS46dt: float = J46
    dS47dt: float = J47
    dS48dt: float = J48
    dS49dt: float = J49
    dS50dt: float = J50
    dS51dt: float = J51
    return (
        dS0dt,
        dS1dt,
        dS10dt,
        dS11dt,
        dS12dt,
        dS13dt,
        dS14dt,
        dS15dt,
        dS16dt,
        dS17dt,
        dS18dt,
        dS19dt,
        dS2dt,
        dS20dt,
        dS21dt,
        dS22dt,
        dS23dt,
        dS24dt,
        dS25dt,
        dS26dt,
        dS27dt,
        dS28dt,
        dS29dt,
        dS3dt,
        dS30dt,
        dS31dt,
        dS32dt,
        dS33dt,
        dS34dt,
        dS35dt,
        dS36dt,
        dS37dt,
        dS38dt,
        dS39dt,
        dS4dt,
        dS40dt,
        dS41dt,
        dS42dt,
        dS43dt,
        dS44dt,
        dS45dt,
        dS46dt,
        dS47dt,
        dS48dt,
        dS49dt,
        dS5dt,
        dS50dt,
        dS51dt,
        dS6dt,
        dS7dt,
        dS8dt,
        dS9dt,
    )


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (
        S0,
        S1,
        S10,
        S11,
        S12,
        S13,
        S14,
        S15,
        S16,
        S17,
        S18,
        S19,
        S2,
        S20,
        S21,
        S22,
        S23,
        S24,
        S25,
        S26,
        S27,
        S28,
        S29,
        S3,
        S30,
        S31,
        S32,
        S33,
        S34,
        S35,
        S36,
        S37,
        S38,
        S39,
        S4,
        S40,
        S41,
        S42,
        S43,
        S44,
        S45,
        S46,
        S47,
        S48,
        S49,
        S5,
        S50,
        S51,
        S6,
        S7,
        S8,
        S9,
    ) = variables
    J0: float = math.e
    J1: float = math.exp(math.e)
    J2: float = 1
    J3: float = 1
    J4: float = math.pi
    J5: float = 1.04719755119660
    J6: float = (1 / 2) * math.pi
    J7: float = -0.523598775598299
    J8: float = 1.22777238637419
    J9: float = -1.43067687253053
    J10: float = 1
    J11: float = 4
    J12: float = -4
    J13: float = -0.947721602131112
    J14: float = 0.975897449330606
    J15: float = 1
    J16: float = math.e
    J17: float = 2.15976625378492
    J18: float = -5
    J19: float = 9
    J20: float = -1.60943791243410
    J21: float = 0
    J22: float = -1.6094379124341 / math.log(10)
    J23: float = 0
    J24: float = 1
    J25: float = 4
    J26: float = 5.56228172775440
    J27: float = 16
    J28: float = 9.61000000000000
    J29: float = 2
    J30: float = 2.72029410174709
    J31: float = 0.863209366648874
    J32: float = 0
    J33: float = 0.373876664830236
    J34: float = 0
    J35: float = 2.01433821447683
    J36: float = -math.tan(6)
    J37: float = 1.13949392732455
    J38: float = -1.02298638367130
    J39: float = 4.93315487558689
    J40: float = 0.304520293447143
    J41: float = 2.82831545788997
    J42: float = 1.12099946637054
    J43: float = 1.14109666064347
    J44: float = -1.47112767430373
    J45: float = math.asinh(99)
    J46: float = 0.802881971289134
    J47: float = -0.867300527694053
    J48: float = 1.51330668842682
    J49: float = 5.29834236561059
    J50: float = 1
    J51: float = 0
    return {
        "J0": J0,
        "J1": J1,
        "J2": J2,
        "J3": J3,
        "J4": J4,
        "J5": J5,
        "J6": J6,
        "J7": J7,
        "J8": J8,
        "J9": J9,
        "J10": J10,
        "J11": J11,
        "J12": J12,
        "J13": J13,
        "J14": J14,
        "J15": J15,
        "J16": J16,
        "J17": J17,
        "J18": J18,
        "J19": J19,
        "J20": J20,
        "J21": J21,
        "J22": J22,
        "J23": J23,
        "J24": J24,
        "J25": J25,
        "J26": J26,
        "J27": J27,
        "J28": J28,
        "J29": J29,
        "J30": J30,
        "J31": J31,
        "J32": J32,
        "J33": J33,
        "J34": J34,
        "J35": J35,
        "J36": J36,
        "J37": J37,
        "J38": J38,
        "J39": J39,
        "J40": J40,
        "J41": J41,
        "J42": J42,
        "J43": J43,
        "J44": J44,
        "J45": J45,
        "J46": J46,
        "J47": J47,
        "J48": J48,
        "J49": J49,
        "J50": J50,
        "J51": J51,
    }
