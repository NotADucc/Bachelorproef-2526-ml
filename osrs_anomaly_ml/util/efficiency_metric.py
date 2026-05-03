from dataclasses import dataclass


@dataclass
class EHPMetric():
    start_lvl: int
    end_lvl: int | None
    value: int


@dataclass
class EHBMetric():
    value: int


EfficiencyMetrics: dict[str, list[EHPMetric] | list[EHBMetric]] = {
    "woodcutting": [
        EHPMetric(0, 15, 29_000),
        EHPMetric(15, 35, 56_000),
        EHPMetric(35, 41, 93_174),
        EHPMetric(41, 51, 114_728),
        EHPMetric(51, 61, 127_339),
        EHPMetric(61, 71, 172_507),
        EHPMetric(71, 80, 194_022),
        EHPMetric(80, 90, 207_636),
        EHPMetric(90, 99, 221_977),
        EHPMetric(99, None, 235_000),
    ],
    "mining": [
        EHPMetric(0, 39, 20_000),
        EHPMetric(39, 63, 50_600),
        EHPMetric(63, 75, 106_540),
        EHPMetric(75, 85, 112_166),
        EHPMetric(85, 95, 116_760),
        EHPMetric(95, 99, 119_438),
        EHPMetric(99, None, 126_000),
    ],
    "thieving": [
        EHPMetric(0, 45, 15_000),
        EHPMetric(45, 49, 80_000),
        EHPMetric(49, 60, 241_906),
        EHPMetric(60, 75, 279_597),
        EHPMetric(75, 88, 333_283),
        EHPMetric(88, 97, 370_532),
        EHPMetric(97, 99, 363_882),
        EHPMetric(99, None, 370_169),
    ],
    "runecrafting": [
        EHPMetric(0, 38, 13_600),
        EHPMetric(38, 75, 45_000),
        EHPMetric(75, 85, 75_400),
        EHPMetric(85, 99, 106_100),
        EHPMetric(99, None, 162_000),
    ],
    "artio": [EHBMetric(60)],
    "callisto": [EHBMetric(142)],
    "calvarion": [EHBMetric(55)],
    "corp": [EHBMetric(60)],
    "dks_rex": [EHBMetric(105)],
    "doom_mokhaiotl": [EHBMetric(20)],
    "nightmare": [EHBMetric(14)],
    "psn": [EHBMetric(9)],
    "spindel": [EHBMetric(55)],
    "cg": [EHBMetric(7)],
    "tob": [EHBMetric(3)],
    "venenatis": [EHBMetric(80)],
    "vetion": [EHBMetric(50)],
    "yama": [EHBMetric(18)],
}


def from_string(s: str):
    key = s.lower()

    try:
        return EfficiencyMetrics[key]
    except KeyError:
        valid_values = ', '.join(EfficiencyMetrics.keys())
        raise KeyError(f"value given: {s}, valid values [{valid_values}]")
