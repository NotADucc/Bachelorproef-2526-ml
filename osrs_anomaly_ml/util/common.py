import json
import os
from collections.abc import Mapping, Sequence
from itertools import product

import pandas as pd
from osrs_hiscore_scrape.request.hs_types import HSType
from pandas import DataFrame


def read_data_dump(file):
    data = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def _is_mapping(x) -> bool:
    return isinstance(x, Mapping)


def _is_sequence(x) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))


def _flatten(obj, parent_key: str = "", sep: str = "_") -> dict:
    items = {}

    if _is_mapping(obj):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten(v, new_key, sep=sep))

    elif _is_sequence(obj):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten(v, new_key, sep=sep))

    else:
        items[parent_key] = obj

    return items


def flatten_json_to_df(data, sep: str = "_") -> pd.DataFrame:
    if _is_sequence(data) and not _is_mapping(data):
        rows = [_flatten(x, sep=sep) for x in data]
    else:
        rows = [_flatten(data, sep=sep)]
    return pd.DataFrame(rows)


def flatten_hiscore_record(data: list[dict]) -> list[dict]:
    """
    Prepare OSRS hiscore JSON list for flattening:
    - Rename 'rank' → 'total_rank'
    - Lift everything under 'record' up (removing 'record_' prefix)
    """
    processed = []
    for row in data:
        new_row = {}

        new_row["category_rank"] = row.get("rank")
        new_row["is_bot"] = row.get("is_bot")

        record = row.get("record", {})
        for key, value in record.items():
            new_row[key] = value

        processed.append(new_row)

    return processed


def build_hstype_lookup() -> dict[str, str]:
    return {
        hs.name: (
            "skill" if hs.is_skill()
            else "boss" if hs.is_boss()
            else "misc"
        )
        for hs in HSType if hs.get_csv_value() != -1
    }


def prefix_columns(df: pd.DataFrame) -> pd.DataFrame:
    hstype_map = build_hstype_lookup()
    new_columns = {}

    for col in df.columns:
        parts = col.split("_")
        renamed = False
        for start in range(len(parts)):
            for end in range(len(parts), start, -1):
                candidate = "_".join(parts[start:end])

                if candidate in hstype_map:
                    prefix = hstype_map[candidate]

                    new_columns[col] = "_".join(
                        [prefix, candidate] + parts[end:]
                    )
                    renamed = True
                    break
            if renamed:
                break

        if not renamed:
            new_columns[col] = col

    return df.rename(columns=new_columns)


def fill_missing_values_df(df: DataFrame) -> DataFrame:
    return df.fillna(0)


def fill_missing_hiscore_data(df: DataFrame) -> DataFrame:
    skill_lvl_cols = [c for c in df.columns if c.startswith(
        "skill_") and c.endswith("_lvl")]
    skill_xp_cols = [c for c in df.columns if c.startswith(
        "skill_") and c.endswith("_xp")]

    if "total_lvl" not in df.columns:
        df["total_lvl"] = None

    mask_lvl = df["total_lvl"].isna() | df["total_lvl"] == 0
    df.loc[mask_lvl, "total_lvl"] = df.loc[mask_lvl,
                                           skill_lvl_cols].sum(axis=1)

    if "total_xp" not in df.columns:
        df["total_xp"] = None

    mask_xp = df["total_xp"].isna() | df["total_xp"] == 0
    df.loc[mask_xp, "total_xp"] = df.loc[mask_xp, skill_xp_cols].sum(axis=1)

    return df


def normalize_number(v):
    try:
        f = float(v)
        if f.is_integer():
            return int(f)
        return f
    except:
        return v


def param_combinations(param_dict):
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in product(*values)]


def evaluate_model(df, labels, score, y, model_name, params):
    results = df.copy()

    for col in y:
        results[col] = y[col].values

    # check if all models follow lower score = anomaly
    results["anomaly_label"] = labels
    results["is_anomaly"] = results["anomaly_label"] == -1

    if score is not None:
        results["anomaly_score"] = score
        results.sort_values("anomaly_score", ascending=True, inplace=True)
    else:
        # fallback: put predicted anomalies first
        results.sort_values("is_anomaly", ascending=False, inplace=True)

    return results


def append_summary_result(write_to, result_dict, sort_by):
    new_row = pd.DataFrame([result_dict])

    if os.path.exists(write_to):
        existing = pd.read_csv(write_to)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    if isinstance(sort_by, dict):
        by = list(sort_by.keys())
        ascending = list(sort_by.values())
    else:
        by = sort_by if isinstance(sort_by, list) else [sort_by]
        ascending = [False] * len(by)

    updated.sort_values(by=by, ascending=ascending, inplace=True)

    updated.to_csv(write_to, index=False)
