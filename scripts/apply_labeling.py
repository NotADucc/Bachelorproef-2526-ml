import argparse
import asyncio
import json
import os

from osrs_anomaly_ml.util.benchmarking import benchmark
from osrs_anomaly_ml.util.common import read_data_dump
from osrs_anomaly_ml.util.log import get_logger, log_execution

logger = get_logger(__name__)


@log_execution
@benchmark
async def main(in_file: str, out_file: str, users_file: str, score: float):
    logger.debug("loading labels")
    with open(users_file, "r", encoding="utf-8") as f:
        target_usernames = {line.strip().lower() for line in f if line.strip()}
    logger.debug(f"label lines found: {len(target_usernames)}")

    logger.debug("loading json lines")
    data_json = read_data_dump(in_file)
    logger.debug(f"json loaded: {len(data_json)}")

    logger.debug(f'apply scoring')
    found = 0
    for row in data_json:
        username = row["record"]["username"].lower()
        if username in target_usernames:
            row["is_bot"] = score
            found = found + 1
            target_usernames.remove(username)
        else:
            row["is_bot"] = row.get("is_bot", 0)

    logger.debug(
        f'scoring applied, found {found}, missing {len(target_usernames)}')
    logger.debug(target_usernames)  # type: ignore

    logger.debug(f'start write')
    with open(out_file, mode='w') as f:
        for row in data_json:
            f.write(json.dumps(row) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', required=True,
                        help="Path to the input JSON file. Each line must be a JSON object containing a record.username field.")
    parser.add_argument('--out-file', required=True,
                        help="Path to the output file.")
    parser.add_argument('--users-file', required=True,
                        help="Path to a text file containing usernames (one per line). Matching is case-insensitive.")
    parser.add_argument('--score', required=True,
                        help="Numeric label assigned to matching users")
    args = parser.parse_args()

    asyncio.run(main(args.in_file, args.out_file, args.users_file, float(args.score)))
