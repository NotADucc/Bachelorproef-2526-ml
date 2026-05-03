import argparse
import asyncio
import csv
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from osrs_anomaly_ml.util import features
from osrs_anomaly_ml.util.benchmarking import benchmark
from osrs_anomaly_ml.util.common import (append_summary_result, fill_missing_values_df,
                                         flatten_json_to_df, normalize_number,
                                         param_combinations,
                                         fill_missing_hiscore_data,
                                         prefix_columns,
                                         flatten_hiscore_record,
                                         read_data_dump)
from osrs_anomaly_ml.util.log import get_logger, log_execution
from osrs_anomaly_ml.util.models import (run_dbscan, run_elliptic,
                                         run_isolation_forest, run_lof,
                                         run_ocsvm)

PCA_DEFAULT_SIZE = 30
RANDOM_STATE = 42
SUMMARY_NAME = "summary_results.csv"
logger = get_logger(__name__)


@log_execution
@benchmark
async def main(in_file: str, output_dir: str, settings_file: str, is_summary_mode: bool):
    logger.debug('reading json')
    data_json = read_data_dump(in_file)
    logger.debug('json loaded')

    with open(settings_file, "r") as f:
        settings = json.load(f)

    logger.debug('settings read')

    in_file_PATH = pathlib.Path(in_file)
    in_file_name = in_file_PATH.stem
    os.makedirs(output_dir, exist_ok=True)
    logger.debug('output dir made')

    # rename rank to total rank to remove confusion with category rank
    # flatten 'record' json 'property'
    logger.debug('flatten json record')
    data = flatten_hiscore_record(data_json)

    logger.debug('flatten json into df')
    df = flatten_json_to_df(data, sep="_")

    logger.debug("before prefix features: " + ", ".join(sorted(df.columns)))

    logger.debug('prefix columns')
    df = prefix_columns(df)

    logger.debug("after prefix features: " + ", ".join(sorted(df.columns)))

    logger.debug('fill missing values df')
    df = fill_missing_values_df(df)
    # todo: add missing skills, misc and bosses
    # dataset is large enough atm that there are no missing values
    df = fill_missing_hiscore_data(df)

    logger.debug('normalize numeric')
    df = df.map(normalize_number)

    # can maybe keep combat lvl float
    float_cols = df.select_dtypes(include="float64").columns
    float_cols = float_cols.drop("is_bot", errors="ignore")
    df[float_cols] = df[float_cols].astype("int64")

    df_metadata = df[["is_bot", "username"]].copy()
    df = df.set_index("username")

    logger.debug('Start FE')
    logger.debug('FE: add ratios')
    df = features.add_username_ratio(df)  # WOP
    df = features.add_combat_ratio_lvl(df)
    df = features.add_combat_ratio_xp(df)
    df = features.add_skill_ratios(df)
    df = features.add_boss_ratios(df)

    logger.debug('FE: add patterns')
    df = features.add_scurrius_corp_pattern(df)

    logger.debug('FE: log transform skill')
    df = features.transform_skill_xp_to_log(df)

    logger.debug('FE: Apply weights')
    weights = {
        # "combat_ratio_lvl_missing_flag": 1,
    }
    df = features.apply_feature_weights(df, weights)

    logger.debug('FE: Add specific non linear transformation')
    df = features.add_soft_features(
        df,
        cols=[
            "xp_per_lvl",
            "construction_vs_other_noncombat_xp_ratio",
            "non_combat_xp",
            "total_lvl",
        ],
        method="log1p",
        prefix="log_",
    )

    logger.debug('FS: Remove features')
    df = features.remove_total_category_ranks(df)
    # skills
    df = features.remove_lvl(df, "skill_")
    df = features.remove_xp(df, "skill_")
    df = features.remove_rank(df, "skill_")

    df = features.remove_lvl(df, "log_skill_")
    df = features.remove_xp(df, "log_skill_")
    df = features.remove_rank(df, "log_skill_")

    df = features.remove_xp(df, "total_")
    df = features.remove_xp(df, "log_total")
    df = features.remove_xp(df, "log_combat")

    # misc
    df = features.remove_rank(df, "misc_")
    df = features.remove_kc(df, "misc_")

    # boss
    df = features.remove_rank(df, "boss_")
    df = features.remove_rank(df, "log_boss")
    df = features.remove_kc(df, "boss_")
    df = features.remove_kc(df, "log_boss")

    # extra
    df = features.remove(df, [
        "combat_lvl_value",
        "combat_ratio_lvl_missing_flag",
        "combat_ratio_xp_missing_flag",
        "total_boss_kc",
        "top1_boss_kc",
        "top2_boss_kc_sum",
        "total_lvl",
        "xp_per_lvl",
        "non_combat_xp",
        "construction_vs_other_noncombat_xp_ratio",
    ])

    df = df.sort_index(axis=1)

    # Numeric only
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    logger.debug('FE: remove is bot flag from numeric')
    df_numeric = features.remove(df_numeric, ["is_bot"])

    logger.debug('Finished FE')
    logger.debug("len: " + str(len(df.columns)))
    logger.debug("Engineered features: " + ", ".join(df.columns))
    logger.debug("numeric features: " + ", ".join(df_numeric.columns))

    # df_numeric.to_csv(os.path.join(output_dir, "dump.csv"), index=False)
    # return

    model_runners = {
        "isolation_forest": run_isolation_forest,
        "lof": run_lof,
        "oneclass_svm": run_ocsvm,
        "elliptic_envelope": run_elliptic,
        "elliptic_envelope_pca": run_elliptic,
        "dbscan": run_dbscan,
        "dbscan_pca": run_dbscan,
    }

    logger.debug('create StandardScaler with float32')
    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric).astype(np.float32)

    # simplify dimensions
    logger.debug('create PCA with float32')
    max_components = min(X.shape[0], X.shape[1])
    n_components = min(PCA_DEFAULT_SIZE, max_components)

    # remove pca if feature space is small enough
    # else remove non pca EE since you'll get an error saying feature is too large
    # note: it's still usefull to keep non pca dbscan for hyper param tuning
    if n_components < PCA_DEFAULT_SIZE:
        logger.warning(
            f"Reducing PCA components from {PCA_DEFAULT_SIZE} to {n_components} due to data shape {X.shape}"
        )
        model_runners = {
            k: v for k, v in model_runners.items()
            if "pca" not in k
        }
    else:
        model_runners = {
            k: v for k, v in model_runners.items()
            if k != "elliptic_envelope"
        }

    logger.debug(
        f"initial models filter based on feature count: {model_runners.keys()}")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    PCA_X = pca.fit_transform(X).astype(np.float32)

    model_combinations = 0
    skipped_combinations = 0

    summary_file_path = os.path.join(output_dir, SUMMARY_NAME)
    completed = set()

    if is_summary_mode:
        if os.path.exists(summary_file_path):
            with open(summary_file_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["model_name"], row["params"])
                    completed.add(key)
    else:
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if not filename.endswith(".csv"):
                    continue

                name = filename[:-4]
                matched = False

                for model_name in model_runners.keys():

                    marker = f"_{model_name}_"

                    if marker not in name:
                        continue

                    _, remainder = name.split(marker, 1)
                    params = remainder

                    key = (model_name, params)
                    completed.add(key)
 
                    matched = True
                    break
                if not matched:
                    logger.warning(f"Could not parse filename: {filename}")

    logger.debug('create model combinations')

    all_tasks = []
    for model_name, runner in model_runners.items():
        if model_name not in settings:
            continue

        combos = param_combinations(settings[model_name])
        model_combinations += len(combos)
        for params in combos:
            param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
            key = (model_name, param_str)

            if key in completed:
                skipped_combinations += 1
                continue
            all_tasks.append((model_name, runner, params))

    logger.debug(f'found {model_combinations} model combinations')
    logger.debug(f'skipped {skipped_combinations} model combinations')

    logger.debug('run models sequentially')

    pbar = tqdm(all_tasks, total=len(all_tasks), smoothing=0.01)

    for model_name, runner, params in pbar:
        try:
            param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))

            pbar.set_description(f"{model_name}")
            pbar.set_postfix({"params": param_str[:30]})
            tqdm.write(f"START {model_name}: {param_str}")

            start_time = time.time()
            results: DataFrame = runner(
                X=PCA_X if "pca" in model_name.lower() else X,
                y=df_metadata,
                df=df_numeric,
                model_name=model_name,
                random_state=RANDOM_STATE,
                params=params
            )
            duration = round(time.time() - start_time, 2)

            tqdm.write(
                f"DONE TRAINING {model_name}: {param_str} | {duration} secs")

            outlier_count = results["is_anomaly"].value_counts()
            false_count = int(outlier_count.get(False, 0))
            true_count = int(outlier_count.get(True, 0))

            result_message = f"{model_name}: {param_str}, " \
                + f"anomaly count: False={false_count}, True={true_count}"

            if not is_summary_mode:
                filename = f"{in_file_name}_{model_name}_{param_str}.csv"
                results.to_csv(os.path.join(output_dir, filename), index=False)
            else:
                if "is_bot" not in results.columns:
                    raise ValueError(
                        "Column 'is_bot' is required for summary scoring")

                summary = {
                    "model_name": model_name,
                    "params": param_str,
                    "time_taken_seconds": duration,
                    "anomaly_count": true_count,
                }

                top_ks = (50, 100, 200)
                for k in top_ks:
                    effective_k = k if k <= true_count else true_count
                    top_effective_k = results.head(effective_k)
                    top_k = results.head(k)
                    summary.update({
                        f"top_{k}": top_effective_k["is_bot"].sum(),
                        f"mean_{k}": top_effective_k["is_bot"].mean(),
                        f"std_{k}": top_effective_k["is_bot"].std(),
                        f"top_{k}_without_pentalty": (top_k["is_bot"] > 0).sum(),
                        f"mean_{k}_without_pentalty": (top_k["is_bot"] > 0).mean(),
                        f"std_{k}_without_pentalty": (top_k["is_bot"] > 0).std(),
                    })

                append_summary_result(
                    write_to=summary_file_path,
                    result_dict=summary,
                    sort_by={"top_100": False, "anomaly_count": True}
                )

            tqdm.write(f"PROCESSED RESULT {result_message}")

        except Exception as e:
            logger.error(f"Model failed: {model_name} | {e}")


if __name__ == '__main__':
    def str2mode(v: str) -> str:
        v = v.lower()
        if v not in ("full", "summary"):
            raise argparse.ArgumentTypeError(
                "mode must be 'full' or 'summary'")
        return v

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', required=True,
                        help="Path to the hiscore data")
    parser.add_argument('--out-dir', required=True,
                        help="Directory where results will be written")
    parser.add_argument('--settings-file', required=True,
                        help="Path to the JSON file containing model parameters and combinations")
    parser.add_argument('--mode', type=str2mode, required=True,
                        help="Output mode: `full` (per-model CSV output) or `summary` (aggregated evaluation results)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.in_file, args.out_dir,
                    args.settings_file, args.mode == 'summary'))
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(str(e))
        sys.exit(2)
