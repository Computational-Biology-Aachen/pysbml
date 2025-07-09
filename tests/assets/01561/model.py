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
J0 = 1
J1 = 1
J2 = 1
J3 = 1
J4 = 1
J5 = 1
J6 = 1
J7 = 1
J8 = 1
J9 = 1
J10 = 1
J11 = 1
J12 = 1
J13 = 1
J14 = 1
J15 = 1
J16 = 1
J17 = 1
J18 = 1
J19 = 1
J20 = 1
J21 = 1
J22 = 1
J23 = 1
J24 = 1
J25 = 1
J26 = 1
J27 = 1
J28 = 1
J29 = 1
J30 = 1
J31 = 1
J32 = 1
J33 = 1
J34 = 1
J35 = 1
J36 = 1
J37 = 1
J38 = 1
J39 = 1
J40 = 1
J41 = 1
J42 = 1
J43 = 1
J44 = 1
J45 = 1
J46 = 1
J47 = 1
J48 = 1
J49 = 1
J50 = 1
J51 = 1
S0_stoich = math.e
S1_stoich = math.exp(math.e)
S2_stoich = 1
S3_stoich = 1
S4_stoich = math.pi
S5_stoich = 1.04719755119660
S6_stoich = (1 / 2) * math.pi
S7_stoich = -0.523598775598299
S8_stoich = 1.22777238637419
S9_stoich = -1.43067687253053
S10_stoich = 1
S11_stoich = 4
S12_stoich = -4
S13_stoich = -0.947721602131112
S14_stoich = 0.975897449330606
S15_stoich = 1
S16_stoich = math.e
S17_stoich = 2.15976625378492
S18_stoich = -5
S19_stoich = 9
S20_stoich = -1.60943791243410
S21_stoich = 0
S22_stoich = -1.6094379124341 / math.log(10)
S23_stoich = 0
S24_stoich = 1
S25_stoich = 4
S26_stoich = 5.56228172775440
S27_stoich = 16
S28_stoich = 9.61000000000000
S29_stoich = 2
S30_stoich = 2.72029410174709
S31_stoich = 0.863209366648874
S32_stoich = 0
S33_stoich = 0.373876664830236
S34_stoich = 0
S35_stoich = 2.01433821447683
S36_stoich = -math.tan(6)
S37_stoich = 1.13949392732455
S38_stoich = -1.02298638367130
S39_stoich = 4.93315487558689
S40_stoich = 0.304520293447143
S41_stoich = 2.82831545788997
S42_stoich = 1.12099946637054
S43_stoich = 1.14109666064347
S44_stoich = -1.47112767430373
S45_stoich = math.asinh(99)
S46_stoich = 0.802881971289134
S47_stoich = -0.867300527694053
S48_stoich = 1.51330668842682
S49_stoich = 5.29834236561059
S50_stoich = 1
S51_stoich = 0
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
    J0: float = 1
    J1: float = 1
    J2: float = 1
    J3: float = 1
    J4: float = 1
    J5: float = 1
    J6: float = 1
    J7: float = 1
    J8: float = 1
    J9: float = 1
    J10: float = 1
    J11: float = 1
    J12: float = 1
    J13: float = 1
    J14: float = 1
    J15: float = 1
    J16: float = 1
    J17: float = 1
    J18: float = 1
    J19: float = 1
    J20: float = 1
    J21: float = 1
    J22: float = 1
    J23: float = 1
    J24: float = 1
    J25: float = 1
    J26: float = 1
    J27: float = 1
    J28: float = 1
    J29: float = 1
    J30: float = 1
    J31: float = 1
    J32: float = 1
    J33: float = 1
    J34: float = 1
    J35: float = 1
    J36: float = 1
    J37: float = 1
    J38: float = 1
    J39: float = 1
    J40: float = 1
    J41: float = 1
    J42: float = 1
    J43: float = 1
    J44: float = 1
    J45: float = 1
    J46: float = 1
    J47: float = 1
    J48: float = 1
    J49: float = 1
    J50: float = 1
    J51: float = 1
    dS0dt: float = J0 * S0_stoich
    dS1dt: float = J1 * S1_stoich
    dS2dt: float = J2 * S2_stoich
    dS3dt: float = J3 * S3_stoich
    dS4dt: float = J4 * S4_stoich
    dS5dt: float = J5 * S5_stoich
    dS6dt: float = J6 * S6_stoich
    dS7dt: float = J7 * S7_stoich
    dS8dt: float = J8 * S8_stoich
    dS9dt: float = J9 * S9_stoich
    dS10dt: float = J10 * S10_stoich
    dS11dt: float = J11 * S11_stoich
    dS12dt: float = J12 * S12_stoich
    dS13dt: float = J13 * S13_stoich
    dS14dt: float = J14 * S14_stoich
    dS15dt: float = J15 * S15_stoich
    dS16dt: float = J16 * S16_stoich
    dS17dt: float = J17 * S17_stoich
    dS18dt: float = J18 * S18_stoich
    dS19dt: float = J19 * S19_stoich
    dS20dt: float = J20 * S20_stoich
    dS21dt: float = J21 * S21_stoich
    dS22dt: float = J22 * S22_stoich
    dS23dt: float = J23 * S23_stoich
    dS24dt: float = J24 * S24_stoich
    dS25dt: float = J25 * S25_stoich
    dS26dt: float = J26 * S26_stoich
    dS27dt: float = J27 * S27_stoich
    dS28dt: float = J28 * S28_stoich
    dS29dt: float = J29 * S29_stoich
    dS30dt: float = J30 * S30_stoich
    dS31dt: float = J31 * S31_stoich
    dS32dt: float = J32 * S32_stoich
    dS33dt: float = J33 * S33_stoich
    dS34dt: float = J34 * S34_stoich
    dS35dt: float = J35 * S35_stoich
    dS36dt: float = J36 * S36_stoich
    dS37dt: float = J37 * S37_stoich
    dS38dt: float = J38 * S38_stoich
    dS39dt: float = J39 * S39_stoich
    dS40dt: float = J40 * S40_stoich
    dS41dt: float = J41 * S41_stoich
    dS42dt: float = J42 * S42_stoich
    dS43dt: float = J43 * S43_stoich
    dS44dt: float = J44 * S44_stoich
    dS45dt: float = J45 * S45_stoich
    dS46dt: float = J46 * S46_stoich
    dS47dt: float = J47 * S47_stoich
    dS48dt: float = J48 * S48_stoich
    dS49dt: float = J49 * S49_stoich
    dS50dt: float = J50 * S50_stoich
    dS51dt: float = J51 * S51_stoich
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
    J0: float = 1
    J1: float = 1
    J2: float = 1
    J3: float = 1
    J4: float = 1
    J5: float = 1
    J6: float = 1
    J7: float = 1
    J8: float = 1
    J9: float = 1
    J10: float = 1
    J11: float = 1
    J12: float = 1
    J13: float = 1
    J14: float = 1
    J15: float = 1
    J16: float = 1
    J17: float = 1
    J18: float = 1
    J19: float = 1
    J20: float = 1
    J21: float = 1
    J22: float = 1
    J23: float = 1
    J24: float = 1
    J25: float = 1
    J26: float = 1
    J27: float = 1
    J28: float = 1
    J29: float = 1
    J30: float = 1
    J31: float = 1
    J32: float = 1
    J33: float = 1
    J34: float = 1
    J35: float = 1
    J36: float = 1
    J37: float = 1
    J38: float = 1
    J39: float = 1
    J40: float = 1
    J41: float = 1
    J42: float = 1
    J43: float = 1
    J44: float = 1
    J45: float = 1
    J46: float = 1
    J47: float = 1
    J48: float = 1
    J49: float = 1
    J50: float = 1
    J51: float = 1
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
