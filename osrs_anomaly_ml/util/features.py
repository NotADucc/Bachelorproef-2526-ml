import numpy as np
import pandas
from osrs_hiscore_scrape.request.hs_types import HSType
from pandas import DataFrame

from .log import get_logger

logger = get_logger(__name__)

COMBAT_SKILLS = [
    "attack", "strength", "defence",
    "hitpoints", "ranged", "prayer", "magic"
]
COMBAT_XP_COLS = [f"skill_{skill}_xp" for skill in COMBAT_SKILLS]
COMBAT_LVL_COLS = [f"skill_{skill}_lvl" for skill in COMBAT_SKILLS]

SMALL_NUMBER = 1e-9


# how evenly is it distributed
def _row_entropy(row):
    total = row.sum()
    if total == 0:
        return 0
    probs = row / total
    return -np.sum(probs * np.log(probs + SMALL_NUMBER))


def remove(df: DataFrame, columns: list):
    return df.drop(columns=columns, errors="ignore")


def remove_total_category_ranks(df: DataFrame):
    return remove(df, ["rank", "category_rank", "overall_rank"])


def remove_lvl(df: DataFrame, prefix: str):
    df = df.loc[:, ~(
        df.columns.str.startswith(prefix) &
        df.columns.str.endswith("_lvl")
    )]

    return df


def remove_xp(df: DataFrame, prefix: str):
    df = df.loc[:, ~(
        df.columns.str.startswith(prefix) &
        df.columns.str.endswith("_xp")
    )]

    return df


def remove_rank(df: DataFrame, prefix: str):
    df = df.loc[:, ~(
        df.columns.str.startswith(prefix) &
        df.columns.str.endswith("_rank")
    )]

    return df


def remove_kc(df: DataFrame, prefix: str):
    df = df.loc[:, ~(
        df.columns.str.startswith(prefix) &
        df.columns.str.endswith("_kc")
    )]

    return df


def transform_skill_xp_to_log(df: DataFrame):
    xp_cols = [c for c in df.columns if c.startswith(
        "skill_") and c.endswith("_xp")]
    log_cols = ["log_" + c for c in xp_cols]

    xp_data = df[xp_cols].clip(lower=0)

    log_data = np.log1p(xp_data)
    log_data = log_data.replace([np.inf, -np.inf], 0).fillna(0)  # type: ignore
    log_data.columns = log_cols

    # Concatenate all at once
    df = pandas.concat([df, log_data], axis=1)

    return df


def add_skill_ratios(df: DataFrame):
    skill_xp_cols = [c for c in df.columns if c.startswith(
        "skill_") and c.endswith("_xp")]

    df["skill_diversity_count"] = (df[skill_xp_cols] > 0).sum(axis=1)
    df["xp_per_lvl"] = (
        df["total_xp"] / (df["total_lvl"] + SMALL_NUMBER)).clip(lower=0)

    # probably need to do similar thing like boss count
    # currently its good enough since the dataset is big enough that every
    # skill is most likely in it
    valid_skills = (df[skill_xp_cols] > 0)
    valid_count = valid_skills.sum(axis=1) + 1

    df["xp_entropy"] = df[skill_xp_cols].apply(_row_entropy, axis=1)
    # 40k is around lvl 40, 2m is around lvl 80
    df["low_skill_xp_ratio"] = (
        ((df[skill_xp_cols] < 40_000) & valid_skills).sum(axis=1) / valid_count)
    df["high_skill_xp_ratio"] = (
        ((df[skill_xp_cols] >= 2_000_000) & valid_skills).sum(axis=1) / valid_count)

    non_combat_cols = [
        c for c in skill_xp_cols
        if not any(skill in c for skill in COMBAT_XP_COLS)
    ]

    non_combat_xp = df[non_combat_cols].sum(axis=1)
    df["non_combat_xp"] = non_combat_xp.clip(lower=0)

    non_combat_xp_delta_construction = non_combat_xp - \
        df["skill_construction_xp"]
    df["construction_vs_other_noncombat_xp_ratio"] = (
        df["skill_construction_xp"] / (non_combat_xp_delta_construction + SMALL_NUMBER)).clip(lower=0)

    return df


def add_combat_ratio_lvl(df: DataFrame):
    combat_total_lvl = df[COMBAT_LVL_COLS].sum(axis=1)

    total_lvl_safe = df["total_lvl"].replace(0, np.nan)

    df["combat_ratio_lvl"] = combat_total_lvl / total_lvl_safe
    df["combat_ratio_lvl"] = df["combat_ratio_lvl"].clip(0, 1)
    df["combat_ratio_lvl_missing_flag"] = total_lvl_safe.isna().astype(int)
    df["combat_ratio_lvl"] = df["combat_ratio_lvl"].fillna(0)

    return df


def add_combat_ratio_xp(df: DataFrame):
    combat_total_xp = df[COMBAT_XP_COLS].sum(axis=1)

    total_xp_safe = df["total_xp"].replace(0, np.nan)

    df["combat_ratio_xp"] = combat_total_xp / total_xp_safe
    df["combat_ratio_xp"] = df["combat_ratio_xp"].clip(0, 1)
    df["combat_ratio_xp_missing_flag"] = total_xp_safe.isna().astype(int)
    df["combat_ratio_xp"] = df["combat_ratio_xp"].fillna(0)

    return df


def add_boss_ratios(df: DataFrame):
    boss_kc_cols = [col for col in df.columns if col.startswith(
        "boss_") and col.endswith("_kc")]

    total_boss_kc = df[boss_kc_cols].sum(axis=1)
    nonzero = (df[boss_kc_cols] > 0).sum(axis=1)

    # len(boss_kc_cols) is technically incorrect since you now assume
    # that every boss is included in the dataset
    # to begin with the dataset is a sort of "cross join"
    # even if it contained all the data, u assume that no columns were removed
    # during the FE process
    bosses_len = len([item for item in HSType if item.is_boss()])

    df = df.assign(
        total_boss_kc=total_boss_kc,
        nonzero_bosses=nonzero,
        boss_sparsity=nonzero / bosses_len,
        top1_boss_kc=df[boss_kc_cols].max(axis=1)
    )

    df = df.assign(
        top1_boss_ratio=df["top1_boss_kc"] /
        (df["total_boss_kc"] + SMALL_NUMBER)
    )

    top2_boss = np.sort(df[boss_kc_cols].values, axis=1)[:, -2:]
    df = df.assign(
        top2_boss_kc_sum=top2_boss.sum(axis=1),
        top2_boss_ratio=top2_boss.sum(
            axis=1) / (df["total_boss_kc"] + SMALL_NUMBER),
        kc_entropy=df[boss_kc_cols].apply(_row_entropy, axis=1),
        sub2_bosses_flag=(nonzero <= 2).astype(int)
    )

    log_data = np.log1p(df[boss_kc_cols].clip(lower=0))
    log_data.columns = [f"log_{col}" for col in boss_kc_cols]  # type: ignore

    df = pandas.concat([df, log_data], axis=1)  # type: ignore

    return df


def add_scurrius_corp_pattern(df: DataFrame):
    if "boss_scurrius_kc" in df.columns and "boss_corp_kc" in df.columns:
        df["scurrius_corp_pattern"] = (
            (df["boss_scurrius_kc"] > 500) &
            (df["boss_corp_kc"] > 0) &
            (df["nonzero_bosses"] <= 3)
        ).astype(int)

    return df


def add_username_ratio(df: DataFrame):
    return df


def apply_feature_weights(df: DataFrame, weights: dict, strict: bool = False):
    missing_cols = []

    for col, w in weights.items():
        if col in df.columns:
            df[col] = df[col] * w
        else:
            missing_cols.append(col)

    if missing_cols:
        msg = f"FE: weights skipped, columns not found: {missing_cols}"

        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return df


def add_soft_features(
    df,
    cols: list,
    method: str = "log1p",
    prefix: str = "log_",
    strict: bool = False
):
    df = df.copy()
    missing_cols = []
    created_cols = []

    for col in cols:
        if col not in df.columns:
            missing_cols.append(col)
            continue

        if method == "log1p":
            df[f"{prefix}{col}"] = np.log1p(df[col])

        elif method == "sqrt":
            df[f"{prefix}{col}"] = np.sqrt(df[col])

        elif method == "square":
            df[f"{prefix}{col}"] = df[col] ** 2

        elif method == "sigmoid":
            df[f"{prefix}{col}"] = 1 / (1 + np.exp(-df[col]))

        elif method == "tanh":
            df[f"{prefix}{col}"] = np.tanh(df[col])

        else:
            raise ValueError(f"Unknown method: {method}")

        created_cols.append(f"{prefix}{col}")

    if created_cols:
        logger.info(f"Created soft features: {created_cols}")

    if missing_cols:
        msg = f"Soft feature columns not found: {missing_cols}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return df
