time: float = 0.0
pp8_mrna_degradation_rate: float = 1.00000000000000
pp9_mrna_degradation_rate: float = 1.00000000000000
p1_degradation_rate: float = 0.500000000000000
p2_degradation_rate: float = 0.500000000000000
p3_degradation_rate: float = 0.500000000000000
p4_degradation_rate: float = 0.500000000000000
p5_degradation_rate: float = 0.500000000000000
p6_degradation_rate: float = 0.500000000000000
p7_degradation_rate: float = 0.500000000000000
p8_degradation_rate: float = 0.500000000000000
p9_degradation_rate: float = 0.500000000000000
v1_Kd: float = 11.1470000000000
v1_h: float = 1.00000000000000
v2_Kd: float = 1.00000000000000
v2_h: float = 4.00000000000000
v3_Kd: float = 20.0000000000000
v3_h: float = 1.00000000000000
v4_Kd: float = 0.200000000000000
v4_h: float = 4.00000000000000
v5_Kd: float = 0.200000000000000
v5_h: float = 4.00000000000000
v6_Kd: float = 0.0400000000000000
v6_h: float = 4.00000000000000
v7_Kd: float = 0.0200000000000000
v7_h: float = 4.00000000000000
v8_Kd: float = 0.0400000000000000
v8_h: float = 4.00000000000000
v9_Kd: float = 0.200000000000000
v9_h: float = 4.00000000000000
pp1_mrna_degradation_rate: float = 1.00000000000000
pp2_mrna_degradation_rate: float = 1.00000000000000
pro1_strength: float = 2.00000000000000
pro2_strength: float = 4.50770000000000
pro3_strength: float = 5.00000000000000
pro4_strength: float = 5.00000000000000
pro5_strength: float = 5.00000000000000
pro6_strength: float = 1.31000000000000
pro7_strength: float = 1.31000000000000
pro8_strength: float = 5.00000000000000
pro9_strength: float = 5.00000000000000
pp3_mrna_degradation_rate: float = 1.00000000000000
v10_Kd: float = 0.0200000000000000
v10_h: float = 4.00000000000000
v11_Kd: float = 0.100000000000000
v11_h: float = 2.00000000000000
v12_Kd: float = 0.100000000000000
v12_h: float = 2.00000000000000
v13_Kd: float = 0.0100000000000000
v13_h: float = 2.00000000000000
pp4_mrna_degradation_rate: float = 1.00000000000000
v14_Kd: float = 1.00000000000000
v14_h: float = 4.00000000000000
v15_Kd: float = 20.0000000000000
v15_h: float = 1.00000000000000
pp5_mrna_degradation_rate: float = 1.00000000000000
rbs1_strength: float = 0.366800000000000
rbs2_strength: float = 1.41020000000000
rbs3_strength: float = 0.800000000000000
rbs4_strength: float = 2.21000000000000
rbs5_strength: float = 0.500000000000000
rbs6_strength: float = 2.00000000000000
rbs7_strength: float = 5.00000000000000
rbs8_strength: float = 3.63770000000000
rbs9_strength: float = 8.00000000000000
pp6_mrna_degradation_rate: float = 1.00000000000000
pp7_mrna_degradation_rate: float = 1.00000000000000
DefaultCompartment: float = 1.00000000000000
pp9_mrna: float = 0.0
p9: float = 0.0
pp8_mrna: float = 0.0
p8: float = 0.0
pp7_mrna: float = 0.0
p7: float = 0.0
pp6_mrna: float = 0.0
p6: float = 0.0
pp5_mrna: float = 0.0
p5: float = 0.0
pp4_mrna: float = 0.0
p4: float = 0.0
pp3_mrna: float = 0.0
p3: float = 0.0
pp2_mrna: float = 0.0
p2: float = 0.0
pp1_mrna: float = 0.0
p1: float = 5.00000000000000

# Initial assignments
as8 = (p5 / v9_Kd) ** v9_h / (DefaultCompartment * ((p5 / v9_Kd) ** v9_h + 1))
cod1 = pro1_strength
rs1 = 1.0 / (DefaultCompartment * ((p9 / v13_Kd) ** v13_h + 1))
rs2 = 1.0 / (DefaultCompartment * ((p2 / v2_Kd) ** v2_h + 1))
rs3 = 1.0 / (DefaultCompartment * ((p3 / v3_Kd) ** v3_h + 1))
rs4 = 1.0 / (DefaultCompartment * ((p8 / v11_Kd) ** v11_h + 1))
rs5 = 1.0 / (DefaultCompartment * ((p8 / v12_Kd) ** v12_h + 1))
rs6 = 1.0 / (DefaultCompartment * ((p2 / v14_Kd) ** v14_h + 1))
rs7 = 1.0 / (DefaultCompartment * ((p3 / v15_Kd) ** v15_h + 1))
as1 = (p1 / v1_Kd) ** v1_h / (DefaultCompartment * ((p1 / v1_Kd) ** v1_h + 1))
as2 = (p4 / v4_Kd) ** v4_h / (DefaultCompartment * ((p4 / v4_Kd) ** v4_h + 1))
as3 = (p5 / v5_Kd) ** v5_h / (DefaultCompartment * ((p5 / v5_Kd) ** v5_h + 1))
as4 = (p6 / v6_Kd) ** v6_h / (DefaultCompartment * ((p6 / v6_Kd) ** v6_h + 1))
as5 = (p7 / v7_Kd) ** v7_h / (DefaultCompartment * ((p7 / v7_Kd) ** v7_h + 1))
as6 = (p7 / v10_Kd) ** v10_h / (DefaultCompartment * ((p7 / v10_Kd) ** v10_h + 1))
as7 = (p6 / v8_Kd) ** v8_h / (DefaultCompartment * ((p6 / v8_Kd) ** v8_h + 1))
pp9_mrna_conc = pp9_mrna / DefaultCompartment
p9_conc = p9 / DefaultCompartment
pp8_mrna_conc = pp8_mrna / DefaultCompartment
p8_conc = p8 / DefaultCompartment
pp7_mrna_conc = pp7_mrna / DefaultCompartment
p7_conc = p7 / DefaultCompartment
pp6_mrna_conc = pp6_mrna / DefaultCompartment
p6_conc = p6 / DefaultCompartment
pp5_mrna_conc = pp5_mrna / DefaultCompartment
p5_conc = p5 / DefaultCompartment
pp4_mrna_conc = pp4_mrna / DefaultCompartment
p4_conc = p4 / DefaultCompartment
pp3_mrna_conc = pp3_mrna / DefaultCompartment
p3_conc = p3 / DefaultCompartment
pp2_mrna_conc = pp2_mrna / DefaultCompartment
p2_conc = p2 / DefaultCompartment
pp1_mrna_conc = pp1_mrna / DefaultCompartment
p1_conc = p1 / DefaultCompartment
pp9_v2 = pp9_mrna_conc * pp9_mrna_degradation_rate
pp9_v3 = pp9_mrna_conc * rbs9_strength
pp9_v4 = p9_conc * p9_degradation_rate
pp8_v2 = pp8_mrna_conc * pp8_mrna_degradation_rate
pp8_v3 = pp8_mrna_conc * rbs8_strength
pp8_v4 = p8_conc * p8_degradation_rate
pp7_v2 = pp7_mrna_conc * pp7_mrna_degradation_rate
pp7_v3 = pp7_mrna_conc * rbs7_strength
pp7_v4 = p7_conc * p7_degradation_rate
pp6_v2 = pp6_mrna_conc * pp6_mrna_degradation_rate
pp6_v3 = pp6_mrna_conc * rbs6_strength
pp6_v4 = p6_conc * p6_degradation_rate
pp5_v2 = pp5_mrna_conc * pp5_mrna_degradation_rate
pp5_v3 = pp5_mrna_conc * rbs5_strength
pp5_v4 = p5_conc * p5_degradation_rate
pp4_v2 = pp4_mrna_conc * pp4_mrna_degradation_rate
pp4_v3 = pp4_mrna_conc * rbs4_strength
pp4_v4 = p4_conc * p4_degradation_rate
pp3_v2 = pp3_mrna_conc * pp3_mrna_degradation_rate
pp3_v3 = pp3_mrna_conc * rbs3_strength
pp3_v4 = p3_conc * p3_degradation_rate
pp2_v2 = pp2_mrna_conc * pp2_mrna_degradation_rate
pp2_v3 = pp2_mrna_conc * rbs2_strength
pp2_v4 = p2_conc * p2_degradation_rate
pp1_v1 = cod1
pp1_v2 = pp1_mrna_conc * pp1_mrna_degradation_rate
pp1_v3 = pp1_mrna_conc * rbs1_strength
pp1_v4 = p1_conc * p1_degradation_rate
cod2 = as1 * pro2_strength * rs1
cod3 = pro3_strength * rs2 * rs3
cod4 = pro4_strength * rs6 * rs7
cod5 = as2 * pro5_strength
cod6 = pro6_strength * (as3 + as4)
cod7 = pro7_strength * (as7 + as8)
cod8 = as5 * pro8_strength * rs4
cod9 = as6 * pro9_strength * rs5
pp9_v1 = cod9
pp8_v1 = cod8
pp7_v1 = cod7
pp6_v1 = cod6
pp5_v1 = cod5
pp4_v1 = cod4
pp3_v1 = cod3
pp2_v1 = cod2
y0 = [
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    pp1_mrna,
    pp2_mrna,
    pp3_mrna,
    pp4_mrna,
    pp5_mrna,
    pp6_mrna,
    pp7_mrna,
    pp8_mrna,
    pp9_mrna,
]
variable_names = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "pp1_mrna",
    "pp2_mrna",
    "pp3_mrna",
    "pp4_mrna",
    "pp5_mrna",
    "pp6_mrna",
    "pp7_mrna",
    "pp8_mrna",
    "pp9_mrna",
]


def model(time: float, variables: tuple[float, ...]) -> tuple[float, ...]:
    (
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        pp1_mrna,
        pp2_mrna,
        pp3_mrna,
        pp4_mrna,
        pp5_mrna,
        pp6_mrna,
        pp7_mrna,
        pp8_mrna,
        pp9_mrna,
    ) = variables
    as8: float = (p5 / v9_Kd) ** v9_h / (
        DefaultCompartment * ((p5 / v9_Kd) ** v9_h + 1)
    )
    cod1: float = pro1_strength
    rs1: float = 1.0 / (DefaultCompartment * ((p9 / v13_Kd) ** v13_h + 1))
    rs2: float = 1.0 / (DefaultCompartment * ((p2 / v2_Kd) ** v2_h + 1))
    rs3: float = 1.0 / (DefaultCompartment * ((p3 / v3_Kd) ** v3_h + 1))
    rs4: float = 1.0 / (DefaultCompartment * ((p8 / v11_Kd) ** v11_h + 1))
    rs5: float = 1.0 / (DefaultCompartment * ((p8 / v12_Kd) ** v12_h + 1))
    rs6: float = 1.0 / (DefaultCompartment * ((p2 / v14_Kd) ** v14_h + 1))
    rs7: float = 1.0 / (DefaultCompartment * ((p3 / v15_Kd) ** v15_h + 1))
    as1: float = (p1 / v1_Kd) ** v1_h / (
        DefaultCompartment * ((p1 / v1_Kd) ** v1_h + 1)
    )
    as2: float = (p4 / v4_Kd) ** v4_h / (
        DefaultCompartment * ((p4 / v4_Kd) ** v4_h + 1)
    )
    as3: float = (p5 / v5_Kd) ** v5_h / (
        DefaultCompartment * ((p5 / v5_Kd) ** v5_h + 1)
    )
    as4: float = (p6 / v6_Kd) ** v6_h / (
        DefaultCompartment * ((p6 / v6_Kd) ** v6_h + 1)
    )
    as5: float = (p7 / v7_Kd) ** v7_h / (
        DefaultCompartment * ((p7 / v7_Kd) ** v7_h + 1)
    )
    as6: float = (p7 / v10_Kd) ** v10_h / (
        DefaultCompartment * ((p7 / v10_Kd) ** v10_h + 1)
    )
    as7: float = (p6 / v8_Kd) ** v8_h / (
        DefaultCompartment * ((p6 / v8_Kd) ** v8_h + 1)
    )
    pp9_mrna_conc: float = pp9_mrna / DefaultCompartment
    p9_conc: float = p9 / DefaultCompartment
    pp8_mrna_conc: float = pp8_mrna / DefaultCompartment
    p8_conc: float = p8 / DefaultCompartment
    pp7_mrna_conc: float = pp7_mrna / DefaultCompartment
    p7_conc: float = p7 / DefaultCompartment
    pp6_mrna_conc: float = pp6_mrna / DefaultCompartment
    p6_conc: float = p6 / DefaultCompartment
    pp5_mrna_conc: float = pp5_mrna / DefaultCompartment
    p5_conc: float = p5 / DefaultCompartment
    pp4_mrna_conc: float = pp4_mrna / DefaultCompartment
    p4_conc: float = p4 / DefaultCompartment
    pp3_mrna_conc: float = pp3_mrna / DefaultCompartment
    p3_conc: float = p3 / DefaultCompartment
    pp2_mrna_conc: float = pp2_mrna / DefaultCompartment
    p2_conc: float = p2 / DefaultCompartment
    pp1_mrna_conc: float = pp1_mrna / DefaultCompartment
    p1_conc: float = p1 / DefaultCompartment
    pp9_v2: float = pp9_mrna_conc * pp9_mrna_degradation_rate
    pp9_v3: float = pp9_mrna_conc * rbs9_strength
    pp9_v4: float = p9_conc * p9_degradation_rate
    pp8_v2: float = pp8_mrna_conc * pp8_mrna_degradation_rate
    pp8_v3: float = pp8_mrna_conc * rbs8_strength
    pp8_v4: float = p8_conc * p8_degradation_rate
    pp7_v2: float = pp7_mrna_conc * pp7_mrna_degradation_rate
    pp7_v3: float = pp7_mrna_conc * rbs7_strength
    pp7_v4: float = p7_conc * p7_degradation_rate
    pp6_v2: float = pp6_mrna_conc * pp6_mrna_degradation_rate
    pp6_v3: float = pp6_mrna_conc * rbs6_strength
    pp6_v4: float = p6_conc * p6_degradation_rate
    pp5_v2: float = pp5_mrna_conc * pp5_mrna_degradation_rate
    pp5_v3: float = pp5_mrna_conc * rbs5_strength
    pp5_v4: float = p5_conc * p5_degradation_rate
    pp4_v2: float = pp4_mrna_conc * pp4_mrna_degradation_rate
    pp4_v3: float = pp4_mrna_conc * rbs4_strength
    pp4_v4: float = p4_conc * p4_degradation_rate
    pp3_v2: float = pp3_mrna_conc * pp3_mrna_degradation_rate
    pp3_v3: float = pp3_mrna_conc * rbs3_strength
    pp3_v4: float = p3_conc * p3_degradation_rate
    pp2_v2: float = pp2_mrna_conc * pp2_mrna_degradation_rate
    pp2_v3: float = pp2_mrna_conc * rbs2_strength
    pp2_v4: float = p2_conc * p2_degradation_rate
    pp1_v1: float = cod1
    pp1_v2: float = pp1_mrna_conc * pp1_mrna_degradation_rate
    pp1_v3: float = pp1_mrna_conc * rbs1_strength
    pp1_v4: float = p1_conc * p1_degradation_rate
    cod2: float = as1 * pro2_strength * rs1
    cod3: float = pro3_strength * rs2 * rs3
    cod4: float = pro4_strength * rs6 * rs7
    cod5: float = as2 * pro5_strength
    cod6: float = pro6_strength * (as3 + as4)
    cod7: float = pro7_strength * (as7 + as8)
    cod8: float = as5 * pro8_strength * rs4
    cod9: float = as6 * pro9_strength * rs5
    pp9_v1: float = cod9
    pp8_v1: float = cod8
    pp7_v1: float = cod7
    pp6_v1: float = cod6
    pp5_v1: float = cod5
    pp4_v1: float = cod4
    pp3_v1: float = cod3
    pp2_v1: float = cod2
    dpp9_mrnadt: float = DefaultCompartment * pp9_v1 - DefaultCompartment * pp9_v2
    dp9dt: float = DefaultCompartment * pp9_v3 - DefaultCompartment * pp9_v4
    dpp8_mrnadt: float = DefaultCompartment * pp8_v1 - DefaultCompartment * pp8_v2
    dp8dt: float = DefaultCompartment * pp8_v3 - DefaultCompartment * pp8_v4
    dpp7_mrnadt: float = DefaultCompartment * pp7_v1 - DefaultCompartment * pp7_v2
    dp7dt: float = DefaultCompartment * pp7_v3 - DefaultCompartment * pp7_v4
    dpp6_mrnadt: float = DefaultCompartment * pp6_v1 - DefaultCompartment * pp6_v2
    dp6dt: float = DefaultCompartment * pp6_v3 - DefaultCompartment * pp6_v4
    dpp5_mrnadt: float = DefaultCompartment * pp5_v1 - DefaultCompartment * pp5_v2
    dp5dt: float = DefaultCompartment * pp5_v3 - DefaultCompartment * pp5_v4
    dpp4_mrnadt: float = DefaultCompartment * pp4_v1 - DefaultCompartment * pp4_v2
    dp4dt: float = DefaultCompartment * pp4_v3 - DefaultCompartment * pp4_v4
    dpp3_mrnadt: float = DefaultCompartment * pp3_v1 - DefaultCompartment * pp3_v2
    dp3dt: float = DefaultCompartment * pp3_v3 - DefaultCompartment * pp3_v4
    dpp2_mrnadt: float = DefaultCompartment * pp2_v1 - DefaultCompartment * pp2_v2
    dp2dt: float = DefaultCompartment * pp2_v3 - DefaultCompartment * pp2_v4
    dpp1_mrnadt: float = DefaultCompartment * pp1_v1 - DefaultCompartment * pp1_v2
    dp1dt: float = DefaultCompartment * pp1_v3 - DefaultCompartment * pp1_v4
    return (
        dp1dt,
        dp2dt,
        dp3dt,
        dp4dt,
        dp5dt,
        dp6dt,
        dp7dt,
        dp8dt,
        dp9dt,
        dpp1_mrnadt,
        dpp2_mrnadt,
        dpp3_mrnadt,
        dpp4_mrnadt,
        dpp5_mrnadt,
        dpp6_mrnadt,
        dpp7_mrnadt,
        dpp8_mrnadt,
        dpp9_mrnadt,
    )


def derived(time: float, variables: tuple[float, ...]) -> dict[str, float]:
    (
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        pp1_mrna,
        pp2_mrna,
        pp3_mrna,
        pp4_mrna,
        pp5_mrna,
        pp6_mrna,
        pp7_mrna,
        pp8_mrna,
        pp9_mrna,
    ) = variables
    as8: float = (p5 / v9_Kd) ** v9_h / (
        DefaultCompartment * ((p5 / v9_Kd) ** v9_h + 1)
    )
    cod1: float = pro1_strength
    rs1: float = 1.0 / (DefaultCompartment * ((p9 / v13_Kd) ** v13_h + 1))
    rs2: float = 1.0 / (DefaultCompartment * ((p2 / v2_Kd) ** v2_h + 1))
    rs3: float = 1.0 / (DefaultCompartment * ((p3 / v3_Kd) ** v3_h + 1))
    rs4: float = 1.0 / (DefaultCompartment * ((p8 / v11_Kd) ** v11_h + 1))
    rs5: float = 1.0 / (DefaultCompartment * ((p8 / v12_Kd) ** v12_h + 1))
    rs6: float = 1.0 / (DefaultCompartment * ((p2 / v14_Kd) ** v14_h + 1))
    rs7: float = 1.0 / (DefaultCompartment * ((p3 / v15_Kd) ** v15_h + 1))
    as1: float = (p1 / v1_Kd) ** v1_h / (
        DefaultCompartment * ((p1 / v1_Kd) ** v1_h + 1)
    )
    as2: float = (p4 / v4_Kd) ** v4_h / (
        DefaultCompartment * ((p4 / v4_Kd) ** v4_h + 1)
    )
    as3: float = (p5 / v5_Kd) ** v5_h / (
        DefaultCompartment * ((p5 / v5_Kd) ** v5_h + 1)
    )
    as4: float = (p6 / v6_Kd) ** v6_h / (
        DefaultCompartment * ((p6 / v6_Kd) ** v6_h + 1)
    )
    as5: float = (p7 / v7_Kd) ** v7_h / (
        DefaultCompartment * ((p7 / v7_Kd) ** v7_h + 1)
    )
    as6: float = (p7 / v10_Kd) ** v10_h / (
        DefaultCompartment * ((p7 / v10_Kd) ** v10_h + 1)
    )
    as7: float = (p6 / v8_Kd) ** v8_h / (
        DefaultCompartment * ((p6 / v8_Kd) ** v8_h + 1)
    )
    pp9_mrna_conc: float = pp9_mrna / DefaultCompartment
    p9_conc: float = p9 / DefaultCompartment
    pp8_mrna_conc: float = pp8_mrna / DefaultCompartment
    p8_conc: float = p8 / DefaultCompartment
    pp7_mrna_conc: float = pp7_mrna / DefaultCompartment
    p7_conc: float = p7 / DefaultCompartment
    pp6_mrna_conc: float = pp6_mrna / DefaultCompartment
    p6_conc: float = p6 / DefaultCompartment
    pp5_mrna_conc: float = pp5_mrna / DefaultCompartment
    p5_conc: float = p5 / DefaultCompartment
    pp4_mrna_conc: float = pp4_mrna / DefaultCompartment
    p4_conc: float = p4 / DefaultCompartment
    pp3_mrna_conc: float = pp3_mrna / DefaultCompartment
    p3_conc: float = p3 / DefaultCompartment
    pp2_mrna_conc: float = pp2_mrna / DefaultCompartment
    p2_conc: float = p2 / DefaultCompartment
    pp1_mrna_conc: float = pp1_mrna / DefaultCompartment
    p1_conc: float = p1 / DefaultCompartment
    pp9_v2: float = pp9_mrna_conc * pp9_mrna_degradation_rate
    pp9_v3: float = pp9_mrna_conc * rbs9_strength
    pp9_v4: float = p9_conc * p9_degradation_rate
    pp8_v2: float = pp8_mrna_conc * pp8_mrna_degradation_rate
    pp8_v3: float = pp8_mrna_conc * rbs8_strength
    pp8_v4: float = p8_conc * p8_degradation_rate
    pp7_v2: float = pp7_mrna_conc * pp7_mrna_degradation_rate
    pp7_v3: float = pp7_mrna_conc * rbs7_strength
    pp7_v4: float = p7_conc * p7_degradation_rate
    pp6_v2: float = pp6_mrna_conc * pp6_mrna_degradation_rate
    pp6_v3: float = pp6_mrna_conc * rbs6_strength
    pp6_v4: float = p6_conc * p6_degradation_rate
    pp5_v2: float = pp5_mrna_conc * pp5_mrna_degradation_rate
    pp5_v3: float = pp5_mrna_conc * rbs5_strength
    pp5_v4: float = p5_conc * p5_degradation_rate
    pp4_v2: float = pp4_mrna_conc * pp4_mrna_degradation_rate
    pp4_v3: float = pp4_mrna_conc * rbs4_strength
    pp4_v4: float = p4_conc * p4_degradation_rate
    pp3_v2: float = pp3_mrna_conc * pp3_mrna_degradation_rate
    pp3_v3: float = pp3_mrna_conc * rbs3_strength
    pp3_v4: float = p3_conc * p3_degradation_rate
    pp2_v2: float = pp2_mrna_conc * pp2_mrna_degradation_rate
    pp2_v3: float = pp2_mrna_conc * rbs2_strength
    pp2_v4: float = p2_conc * p2_degradation_rate
    pp1_v1: float = cod1
    pp1_v2: float = pp1_mrna_conc * pp1_mrna_degradation_rate
    pp1_v3: float = pp1_mrna_conc * rbs1_strength
    pp1_v4: float = p1_conc * p1_degradation_rate
    cod2: float = as1 * pro2_strength * rs1
    cod3: float = pro3_strength * rs2 * rs3
    cod4: float = pro4_strength * rs6 * rs7
    cod5: float = as2 * pro5_strength
    cod6: float = pro6_strength * (as3 + as4)
    cod7: float = pro7_strength * (as7 + as8)
    cod8: float = as5 * pro8_strength * rs4
    cod9: float = as6 * pro9_strength * rs5
    pp9_v1: float = cod9
    pp8_v1: float = cod8
    pp7_v1: float = cod7
    pp6_v1: float = cod6
    pp5_v1: float = cod5
    pp4_v1: float = cod4
    pp3_v1: float = cod3
    pp2_v1: float = cod2
    return {
        "as8": as8,
        "cod1": cod1,
        "rs1": rs1,
        "rs2": rs2,
        "rs3": rs3,
        "rs4": rs4,
        "rs5": rs5,
        "rs6": rs6,
        "rs7": rs7,
        "as1": as1,
        "as2": as2,
        "as3": as3,
        "as4": as4,
        "as5": as5,
        "as6": as6,
        "as7": as7,
        "pp9_mrna_conc": pp9_mrna_conc,
        "p9_conc": p9_conc,
        "pp8_mrna_conc": pp8_mrna_conc,
        "p8_conc": p8_conc,
        "pp7_mrna_conc": pp7_mrna_conc,
        "p7_conc": p7_conc,
        "pp6_mrna_conc": pp6_mrna_conc,
        "p6_conc": p6_conc,
        "pp5_mrna_conc": pp5_mrna_conc,
        "p5_conc": p5_conc,
        "pp4_mrna_conc": pp4_mrna_conc,
        "p4_conc": p4_conc,
        "pp3_mrna_conc": pp3_mrna_conc,
        "p3_conc": p3_conc,
        "pp2_mrna_conc": pp2_mrna_conc,
        "p2_conc": p2_conc,
        "pp1_mrna_conc": pp1_mrna_conc,
        "p1_conc": p1_conc,
        "pp9_v2": pp9_v2,
        "pp9_v3": pp9_v3,
        "pp9_v4": pp9_v4,
        "pp8_v2": pp8_v2,
        "pp8_v3": pp8_v3,
        "pp8_v4": pp8_v4,
        "pp7_v2": pp7_v2,
        "pp7_v3": pp7_v3,
        "pp7_v4": pp7_v4,
        "pp6_v2": pp6_v2,
        "pp6_v3": pp6_v3,
        "pp6_v4": pp6_v4,
        "pp5_v2": pp5_v2,
        "pp5_v3": pp5_v3,
        "pp5_v4": pp5_v4,
        "pp4_v2": pp4_v2,
        "pp4_v3": pp4_v3,
        "pp4_v4": pp4_v4,
        "pp3_v2": pp3_v2,
        "pp3_v3": pp3_v3,
        "pp3_v4": pp3_v4,
        "pp2_v2": pp2_v2,
        "pp2_v3": pp2_v3,
        "pp2_v4": pp2_v4,
        "pp1_v1": pp1_v1,
        "pp1_v2": pp1_v2,
        "pp1_v3": pp1_v3,
        "pp1_v4": pp1_v4,
        "cod2": cod2,
        "cod3": cod3,
        "cod4": cod4,
        "cod5": cod5,
        "cod6": cod6,
        "cod7": cod7,
        "cod8": cod8,
        "cod9": cod9,
        "pp9_v1": pp9_v1,
        "pp8_v1": pp8_v1,
        "pp7_v1": pp7_v1,
        "pp6_v1": pp6_v1,
        "pp5_v1": pp5_v1,
        "pp4_v1": pp4_v1,
        "pp3_v1": pp3_v1,
        "pp2_v1": pp2_v1,
    }
