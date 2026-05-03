import argparse
import asyncio

from osrs_hiscore_scrape.request.hs_types import HSType
from osrs_hiscore_scrape.request.records import (PlayerRecordInfo,
                                                 PlayerRecordSkillInfo)
from osrs_hiscore_scrape.util.io import read_player_records

from osrs_anomaly_ml.stats.common import calc_skill_xp
from osrs_anomaly_ml.util import efficiency_metric
from osrs_anomaly_ml.util.benchmarking import benchmark
from osrs_anomaly_ml.util.efficiency_metric import EHBMetric, EHPMetric
from osrs_anomaly_ml.util.log import get_logger, log_execution

logger = get_logger(__name__)


def compute_metric(old: PlayerRecordInfo, new: PlayerRecordInfo, metrics: list[EHPMetric] | list[EHBMetric]):
    if isinstance(metrics[0], EHPMetric):
        assert isinstance(old, PlayerRecordSkillInfo)
        assert isinstance(new, PlayerRecordSkillInfo)

        current_xp = old.xp
        target_xp = new.xp
        ehp = 0.0

        for metric_range in metrics:
            range_start_xp = calc_skill_xp(metric_range.start_lvl)  # type: ignore
            range_end_xp = 200_000_000 if metric_range.end_lvl is None else calc_skill_xp(metric_range.end_lvl)  # type: ignore # nopep8
            if (range_start_xp <= current_xp < range_end_xp):
                boundary = range_end_xp if range_end_xp < target_xp else target_xp
                delta = boundary - current_xp
                ehp += delta / metric_range.value
                current_xp = boundary

            elif current_xp < range_start_xp:
                break

        return ehp

    elif isinstance(metrics[0], EHBMetric):
        delta = new.get_value() - old.get_value()

        kph = metrics[0].value
        return delta / kph

    else:
        raise ValueError("Unknown metric type")


@log_execution
@benchmark
async def main(before_file: str, after_file: str, output_file: str, metric_str: str):
    old_records = list(read_player_records(before_file))
    new_records = list(read_player_records(after_file))
    logger.debug("files read")

    metrics = efficiency_metric.from_string(metric_str)
    hs_type = HSType.from_string(metric_str)
    logger.debug(f"type: {hs_type.name}")

    logger.debug(f"extracting type from records")
    old_dct = {record.username: record.get_stat(
        hs_type) for record in old_records}
    new_dct = {record.username: record.get_stat(
        hs_type) for record in new_records}

    logger.debug(f'start delta calculation')

    results = []
    for username in old_dct:
        if username not in new_dct:
            continue

        old = old_dct[username]
        new = new_dct[username]

        efficiency = compute_metric(old, new, metrics) 

        results.append({
            "key": username,
            "efficiency": efficiency
        })

    logger.debug(f'sorting result')
    results.sort(key=lambda x: x["efficiency"], reverse=True)

    logger.debug(f'writing result to {output_file}')
    with open(output_file, mode="w") as f:
        for item in results:
            f.write(f"{item['key']}: {item['efficiency']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-file', required=True,
                        help="Path to the older snapshot file")
    parser.add_argument('--after-file', required=True,
                        help="Path to the newer snapshot file")
    parser.add_argument('--out-file', required=True,
                        help="Path to the output file")
    parser.add_argument('--hs-type', required=True,
                        help="Hiscore category used to compute efficiency (determines EHP vs EHB)")
    args = parser.parse_args()

    asyncio.run(main(args.before_file, args.after_file,
                args.out_file, args.hs_type))
