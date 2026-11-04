"""전처리 + 집계를 시즌별로 실행하고 data/processed 에 저장.

사용: python -m mlbproj.build   (src 폴더가 PYTHONPATH에 있어야 함; scripts/run_all.py 사용 권장)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import aggregate as A
from .clean import load_and_clean
from .config import GRID_N, GRID_X, GRID_Z, INTERIM, PROCESSED, RAW, SEASONS

INTERIM_TABLES = {"pa", "pitcher_game_velo", "abs_pitcher_edge"}


def _save(name: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "bool":
            pass
    out = (INTERIM if name in INTERIM_TABLES else PROCESSED) / f"{name}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  → {name}.parquet  {len(df):,}행")
    return df


def abs_inputs(df: pd.DataFrame, season: int):
    """ABS 분석용: 월별 리그 지표, 판정 격자, 투수별 경계 의존도."""
    reg = df[~df["is_post"]]
    taken = reg[reg["take"] & reg["description"].isin(["called_strike", "ball", "blocked_ball"])]
    reg = reg.assign(edge_cs=reg["called_strike"] & reg["edge"])
    m = reg.groupby("month").agg(pitches=("swing", "size"),
                                 pa=("pa_end", "sum"), k=("outcome", lambda s: (s == "K").sum()),
                                 bb=("outcome", lambda s: (s == "BB").sum()), edge=("edge", "sum"),
                                 in_zone=("in_zone", "sum")).reset_index()
    tk = taken.groupby("month").agg(taken=("called_strike", "size"), cs=("called_strike", "sum"),
                                    edge_taken=("edge", "sum")).reset_index()
    ed = taken[taken["edge"]].groupby("month")["called_strike"].sum().rename("edge_cs").reset_index()
    oz = taken[~taken["in_zone"]].groupby("month").agg(oz_taken=("called_strike", "size"),
                                                       oz_cs=("called_strike", "sum")).reset_index()
    iz = taken[taken["in_zone"]].groupby("month").agg(iz_taken=("called_strike", "size"),
                                                      iz_cs=("called_strike", "sum")).reset_index()
    m = m.merge(tk, on="month").merge(ed, on="month", how="left").merge(oz, on="month", how="left") \
        .merge(iz, on="month", how="left")
    m["season"] = season
    bx = np.clip(((taken["plate_x"] - GRID_X[0]) / (GRID_X[1] - GRID_X[0]) * GRID_N).astype(int), 0, GRID_N - 1)
    bz = np.clip(((taken["plate_z"] - GRID_Z[0]) / (GRID_Z[1] - GRID_Z[0]) * GRID_N).astype(int), 0, GRID_N - 1)
    grid = pd.DataFrame({"gx": bx, "gz": bz, "cs": taken["called_strike"].astype(int)}) \
        .groupby(["gx", "gz"]).agg(n=("cs", "size"), cs=("cs", "sum")).reset_index()
    grid["season"] = season
    pe = reg.groupby("pitcher").agg(pitches=("edge", "size"), edge=("edge", "sum"), csw=("csw", "sum"),
                                    called=("called_strike", "sum"),
                                    edge_called=("edge_cs", "sum"),
                                    bb=("outcome", lambda s: (s == "BB").sum()), pa=("pa_end", "sum")).reset_index()
    pe["season"] = season
    return m, grid, pe


def pa_table(df: pd.DataFrame, season: int) -> pd.DataFrame:
    pa = df[df["pa_end"]]
    return pd.DataFrame({
        "season": season, "game_date": pa["game_date"].values, "is_post": pa["is_post"].values,
        "pitcher": pa["pitcher"].values, "batter": pa["batter"].values, "stand": pa["stand"].values,
        "p_throws": pa["p_throws"].values, "outcome": pa["outcome"].values,
        "woba": pa["woba_value"].astype("float32").values, "xwoba": pa["xwoba_pa"].astype("float32").values,
        "is_hit": pa["is_hit"].values, "is_ab": pa["is_ab"].values,
    })


def main(seasons=None):
    # data/raw 에 있는 시즌은 모두 사용 (새 시즌만 수집해도 이전 시즌과 함께 다시 집계)
    seasons = seasons or sorted(int(f.stem.split("_")[1]) for f in RAW.glob("statcast_*.parquet"))
    if not seasons:
        raise SystemExit("data/raw 에 statcast_*.parquet 가 없습니다. 먼저 수집을 실행하세요.")
    bucket: dict[str, list] = {}
    put = lambda k, v: bucket.setdefault(k, []).append(v)  # noqa: E731
    for s in seasons:
        t0 = time.time()
        print(f"[build {s}] 정리 중...")
        df = load_and_clean(s)
        print(f"  {len(df):,}구 (정규 {(~df['is_post']).sum():,} / 포스트 {df['is_post'].sum():,})")
        ps = A.pitcher_season(df, s)
        bs = A.batter_season(df, s)
        put("pitcher_season", ps)
        put("batter_season", bs)
        put("pitcher_arsenal", A.pitcher_arsenal(df, s))
        put("pitcher_count_usage", A.count_usage(df, s))
        put("pitcher_splits", A.pitcher_splits(df, s))
        for k, v in A.batter_splits(df, s).items():
            put(f"batter_{k}", v)
        p_ok = ps.loc[ps["qualified"], "pitcher"]
        b_ok = bs.loc[bs["qualified"], "batter"]
        pg, bg = A.location_grids(df, s, p_ok, b_ok)
        put("pitcher_grid", pg)
        put("batter_grid", bg)
        put("batter_spray", A.spray_points(df, s, b_ok))
        put("pitcher_transitions", A.transitions(df, s))
        pm, bm = A.monthly(df, s)
        put("pitcher_monthly", pm)
        put("batter_monthly", bm)
        put("pitcher_game_velo", A.game_velo(df, s))
        pp, bp = A.postseason(df, s)
        put("pitcher_post", pp)
        put("batter_post", bp)
        am, ag, ape = abs_inputs(df, s)
        put("abs_monthly", am)
        put("abs_grid", ag)
        put("abs_pitcher_edge", ape)
        put("pa", pa_table(df, s))
        print(f"  {time.time() - t0:.0f}초")
        del df
    print("[build] 저장")
    for k in list(bucket):
        if k in ("pitcher_season", "batter_season"):
            continue
        _save(k, bucket[k])
    ps = A.add_percentiles(pd.concat(bucket["pitcher_season"], ignore_index=True), A.HIGHER_BETTER_P)
    bs = A.add_percentiles(pd.concat(bucket["batter_season"], ignore_index=True), A.HIGHER_BETTER_B)
    _names(ps, bs)
    _save("pitcher_season", [ps])
    _save("batter_season", [bs])


def _names(ps: pd.DataFrame, bs: pd.DataFrame) -> None:
    """이름 테이블: MLB API 이름 > people > Statcast player_name 순."""
    people_f = RAW / "people.parquet"
    people = pd.read_parquet(people_f) if people_f.exists() else pd.DataFrame(columns=["player_id", "full_name", "birth_date"])
    for t, key in ((ps, "pitcher"), (bs, "batter")):
        nm = t["name_api"] if "name_api" in t else pd.Series(np.nan, index=t.index)
        nm = nm.fillna(t[key].map(people.set_index("player_id")["full_name"]))
        t["name"] = nm.fillna(t[key].astype(str))
        if "birth_date" in people:
            t["birth_date"] = t[key].map(people.set_index("player_id")["birth_date"])
        if "팀" not in t:
            t["팀"] = ""
        t.drop(columns=[c for c in ("name_api",) if c in t], inplace=True)


if __name__ == "__main__":
    main()
