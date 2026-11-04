"""3단계: 선수 해부 리포트용 집계 테이블.

모든 비율은 '분자 합 / 분모 합'으로 계산해 시즌 합산·재집계가 가능하도록 합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXTERNAL, GRID_N, GRID_X, GRID_Z, MIN_PA_BATTER, MIN_PITCHES_PITCHER, RAW


def _sums(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """공통 카운트 합계 (투수·타자 양쪽에서 재사용)."""
    x = df.assign(
        n=1,
        k=df["outcome"].eq("K"), bb=df["outcome"].eq("BB"), hr=df["outcome"].eq("HR"),
        gb=df["bb_type"].eq("ground_ball") & df["bip"], ld=df["bb_type"].eq("line_drive") & df["bip"],
        fb=df["bb_type"].eq("fly_ball") & df["bip"], pu=df["bb_type"].eq("popup") & df["bip"],
        ozone=~df["in_zone"],
        ev_bip=df["launch_speed"].where(df["bip"]), la_bip=df["launch_angle"].where(df["bip"]),
        xba_bip=df["estimated_ba_using_speedangle"].where(df["bip"]),
        xslg_bip=df["estimated_slg_using_speedangle"].where(df["bip"]),
        woba_num=df["woba_value"].where(df["pa_end"]), woba_den=df["woba_denom"].where(df["pa_end"]),
        rv=df["delta_run_exp"],
    )
    cols = ["n", "pa_end", "is_ab", "is_hit", "tb", "k", "bb", "hr", "swing", "whiff", "csw", "called_strike",
            "in_zone", "ozone", "chase", "zone_swing", "contact", "edge", "bip", "hard_hit", "barrel",
            "sweet_spot", "gb", "ld", "fb", "pu", "ev_bip", "la_bip", "xba_bip", "xslg_bip", "xwoba_pa",
            "woba_num", "woba_den", "rv"]
    x["contact_z"] = x["contact"] & x["in_zone"]
    cols.append("contact_z")
    g = x.groupby(keys, observed=True)[cols].sum(min_count=0)
    g["ev_n"] = x.groupby(keys, observed=True)["ev_bip"].count()
    return g.reset_index()


def _rates(g: pd.DataFrame, side: str) -> pd.DataFrame:
    """합계 → 비율 지표. side='pitcher'면 실점 가치를 투수 관점(+가 좋음)으로 뒤집음."""
    s = lambda a, b: np.where(g[b] > 0, g[a] / g[b].replace(0, np.nan), np.nan)  # noqa: E731
    out = g.copy()
    out["pitches"] = g["n"]
    out["PA"] = g["pa_end"]
    out["K%"] = s("k", "pa_end")
    out["BB%"] = s("bb", "pa_end")
    out["K-BB%"] = out["K%"] - out["BB%"]
    out["타율"] = s("is_hit", "is_ab")
    out["장타율"] = s("tb", "is_ab")
    out["wOBA"] = s("woba_num", "woba_den")
    out["xwOBA"] = s("xwoba_pa", "pa_end")
    out["xBA"] = s("xba_bip", "is_ab")
    out["xSLG"] = s("xslg_bip", "is_ab")
    out["Swing%"] = s("swing", "n")
    out["Whiff%"] = s("whiff", "swing")
    out["CSW%"] = s("csw", "n")
    out["Zone%"] = s("in_zone", "n")
    out["Chase%"] = s("chase", "ozone")
    out["Z-Swing%"] = s("zone_swing", "in_zone")
    out["Z-Contact%"] = s("contact_z", "zone_swing")
    out["Edge%"] = s("edge", "n")
    out["평균 타구속도"] = s("ev_bip", "ev_n")
    out["평균 발사각"] = s("la_bip", "ev_n")
    out["HardHit%"] = s("hard_hit", "bip")
    out["Barrel%"] = s("barrel", "bip")
    out["SweetSpot%"] = s("sweet_spot", "bip")
    out["GB%"] = s("gb", "bip")
    out["LD%"] = s("ld", "bip")
    out["FB%"] = s("fb", "bip")
    out["HR"] = g["hr"]
    sign = -1 if side == "pitcher" else 1
    out["RV/100"] = sign * g["rv"] / g["n"].replace(0, np.nan) * 100
    out["RV"] = sign * g["rv"]
    return out


RATE_COLS = ["pitches", "PA", "K%", "BB%", "K-BB%", "타율", "장타율", "wOBA", "xwOBA", "xBA", "xSLG", "Swing%",
             "Whiff%", "CSW%", "Zone%", "Chase%", "Z-Swing%", "Z-Contact%", "Edge%", "평균 타구속도", "평균 발사각",
             "HardHit%", "Barrel%", "SweetSpot%", "GB%", "LD%", "FB%", "HR", "RV/100", "RV"]


def _ip(row) -> float:
    if "outs" in row and pd.notna(row.get("outs")):
        return row["outs"] / 3
    ip = str(row.get("inningsPitched", "0"))
    whole, _, frac = ip.partition(".")
    return int(whole or 0) + int(frac or 0) / 3


def _load_fg(season: int, kind: str) -> pd.DataFrame | None:
    f = EXTERNAL / f"fangraphs_{kind}_{season}.csv"
    if not f.exists():
        return None
    fg = pd.read_csv(f)
    idc = next((c for c in ("MLBAMID", "mlbamid", "key_mlbam", "xMLBAMID") if c in fg.columns), None)
    if idc is None:
        return None
    keep = {idc: "player_id"}
    for c, new in (("WAR", "fWAR"), ("wRC+", "wRC+"), ("xFIP", "xFIP"), ("SIERA", "SIERA"), ("Stuff+", "Stuff+")):
        if c in fg.columns:
            keep[c] = new
    return fg[list(keep)].rename(columns=keep).dropna(subset=["player_id"]).astype({"player_id": int})


def pitcher_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]]
    t = _rates(_sums(reg, ["pitcher"]), "pitcher")[["pitcher"] + RATE_COLS]
    fb = reg[reg["pitch_group"] == "패스트볼"].groupby("pitcher")["release_speed"].agg(["mean", "max"])
    t = t.merge(fb.rename(columns={"mean": "패스트볼 평균구속", "max": "최고구속"}), left_on="pitcher",
                right_index=True, how="left")
    t["구종 수"] = t["pitcher"].map(
        reg.groupby(["pitcher", "pitch_type"]).size().pipe(lambda s: s[s / s.groupby(level=0).transform("sum") >= .03])
        .groupby(level=0).size())
    t["p_throws"] = t["pitcher"].map(reg.groupby("pitcher")["p_throws"].first())
    t["age"] = t["pitcher"].map(reg.groupby("pitcher")["age_pit"].median())
    api_f = RAW / f"mlbapi_pitching_{season}.parquet"
    if api_f.exists():
        api = pd.read_parquet(api_f)
        api["IP"] = api.apply(_ip, axis=1)
        for c in ("earnedRuns", "strikeOuts", "baseOnBalls", "hitByPitch", "homeRuns", "hits", "wins", "losses", "saves", "gamesPlayed", "gamesStarted"):
            if c not in api:
                api[c] = np.nan
        lg_era = 9 * api["earnedRuns"].sum() / api["IP"].sum()
        c_fip = lg_era - (13 * api["homeRuns"].sum() + 3 * (api["baseOnBalls"].sum() + api["hitByPitch"].sum())
                          - 2 * api["strikeOuts"].sum()) / api["IP"].sum()
        api["ERA"] = 9 * api["earnedRuns"] / api["IP"].replace(0, np.nan)
        api["FIP"] = (13 * api["homeRuns"] + 3 * (api["baseOnBalls"] + api["hitByPitch"]) - 2 * api["strikeOuts"]) \
            / api["IP"].replace(0, np.nan) + c_fip
        api["WHIP"] = (api["hits"] + api["baseOnBalls"]) / api["IP"].replace(0, np.nan)
        api = api.rename(columns={"wins": "승", "losses": "패", "saves": "세이브", "gamesPlayed": "경기",
                                  "gamesStarted": "선발", "name": "name_api", "team": "팀"})
        t = t.merge(api[["player_id", "name_api", "팀", "경기", "선발", "IP", "ERA", "FIP", "WHIP", "승", "패", "세이브"]],
                    left_on="pitcher", right_on="player_id", how="left").drop(columns="player_id")
    fg = _load_fg(season, "pitching")
    if fg is not None:
        t = t.merge(fg, left_on="pitcher", right_on="player_id", how="left").drop(columns="player_id")
    t["ERA-FIP"] = t.get("ERA") - t.get("FIP") if "ERA" in t else np.nan
    t["season"] = season
    t["qualified"] = t["pitches"] >= MIN_PITCHES_PITCHER
    return t


def batter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]]
    t = _rates(_sums(reg, ["batter"]), "batter")[["batter"] + RATE_COLS]
    bip = reg[reg["bip"]]
    t["최고 타구속도"] = t["batter"].map(bip.groupby("batter")["launch_speed"].max())
    t["EV90"] = t["batter"].map(bip.groupby("batter")["launch_speed"].quantile(.9))
    sw = reg[reg["swing"] & reg["bat_speed"].notna()]
    t["배트 스피드"] = t["batter"].map(sw.groupby("batter")["bat_speed"].mean())
    t["스윙 길이"] = t["batter"].map(sw.groupby("batter")["swing_length"].mean())
    dirs = bip.dropna(subset=["direction"]).groupby(["batter", "direction"], observed=True).size().unstack(fill_value=0)
    dirs = dirs.div(dirs.sum(axis=1), axis=0)
    for c in ("당겨치기", "가운데", "밀어치기"):
        t[f"{c}%"] = t["batter"].map(dirs[c]) if c in dirs else np.nan
    t["stand"] = t["batter"].map(reg.groupby("batter")["stand"].agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0]))
    t["age"] = t["batter"].map(reg.groupby("batter")["age_bat"].median())
    # 리그 wOBA (정규시즌 전체 타석)
    pa_rows = reg[reg["pa_end"] & reg["woba_denom"].notna()]
    lg_woba = pa_rows["woba_value"].sum() / max(pa_rows["woba_denom"].sum(), 1)
    lg_r_pa = np.nan
    api_f = RAW / f"mlbapi_hitting_{season}.parquet"
    if api_f.exists():
        api = pd.read_parquet(api_f)
        if {"runs", "plateAppearances"} <= set(api.columns):
            lg_r_pa = pd.to_numeric(api["runs"], errors="coerce").sum() / pd.to_numeric(api["plateAppearances"], errors="coerce").sum()
        ren = {"name": "name_api", "team": "팀", "gamesPlayed": "경기", "avg": "AVG", "obp": "OBP", "slg": "SLG",
               "ops": "OPS", "homeRuns": "홈런", "rbi": "타점", "runs": "득점", "stolenBases": "도루", "hits": "안타"}
        api = api.rename(columns=ren)
        for c in ren.values():
            if c not in api:
                api[c] = np.nan
        for c in ("AVG", "OBP", "SLG", "OPS"):
            api[c] = pd.to_numeric(api[c], errors="coerce")
        t = t.merge(api[["player_id"] + list(ren.values())], left_on="batter", right_on="player_id",
                    how="left").drop(columns="player_id")
    fg = _load_fg(season, "batting")
    if fg is not None:
        t = t.merge(fg, left_on="batter", right_on="player_id", how="left").drop(columns="player_id")
    # 자체 wRC+ (FanGraphs 값이 없을 때): 100 × (wRAA/PA + 리그 R/PA) / 리그 R/PA, wOBA 스케일 1.24, 구장 보정 없음
    WOBA_SCALE = 1.24
    if pd.notna(lg_r_pa) and lg_r_pa > 0:
        calc = 100 * (((t["wOBA"] - lg_woba) / WOBA_SCALE) + lg_r_pa) / lg_r_pa
    else:
        calc = pd.Series(np.nan, index=t.index)
    if "wRC+" in t and t["wRC+"].notna().any():
        t["wRC+ 출처"] = np.where(t["wRC+"].notna(), "FanGraphs", "자체 계산")
        t["wRC+"] = t["wRC+"].fillna(calc)
    else:
        t["wRC+"] = calc
        t["wRC+ 출처"] = "자체 계산"
    t["wOBA-xwOBA"] = t["wOBA"] - t["xwOBA"]
    t["season"] = season
    t["qualified"] = t["PA"] >= MIN_PA_BATTER
    return t


def pitcher_arsenal(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]]
    keys = ["pitcher", "pitch_type"]
    t = _rates(_sums(reg, keys), "pitcher")[keys + RATE_COLS]
    phys = reg.groupby(keys).agg(
        구속=("release_speed", "mean"), 최고구속=("release_speed", "max"), 회전수=("release_spin_rate", "mean"),
        회전축=("spin_axis", "mean"), 수직무브=("ivb_in", "mean"), 수평무브=("hb_in", "mean"),
        수직무브_sd=("ivb_in", "std"), 수평무브_sd=("hb_in", "std"), 릴리스높이=("release_pos_z", "mean"),
        릴리스측면=("rel_x_norm", "mean"), 익스텐션=("release_extension", "mean"),
        평균위치x=("plate_x", "mean"), 평균위치z=("plate_z", "mean")).reset_index()
    t = t.merge(phys, on=keys)
    tot = t.groupby("pitcher")["pitches"].transform("sum")
    t["구사율"] = t["pitches"] / tot
    vs = reg.groupby(["pitcher", "stand", "pitch_type"]).size().unstack("stand", fill_value=0)
    vs = vs / vs.groupby(level=0).transform("sum")
    for st, lab in (("L", "좌타 구사율"), ("R", "우타 구사율")):
        t[lab] = t.set_index(keys).index.map(vs[st]) if st in vs else np.nan
    # 구속 차이 (주 패스트볼 대비)
    prim = t[t["pitch_type"].isin(["FF", "SI"])].sort_values("pitches").groupby("pitcher").tail(1).set_index("pitcher")
    t["패스트볼 대비 구속차"] = t["구속"] - t["pitcher"].map(prim["구속"])
    t["수직무브 차"] = t["수직무브"] - t["pitcher"].map(prim["수직무브"])
    t["수평무브 차"] = t["수평무브"] - t["pitcher"].map(prim["수평무브"])
    t["season"] = season
    return t


def count_usage(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]]
    g = reg.groupby(["pitcher", "count_state", "pitch_type"]).size().rename("n").reset_index()
    g["share"] = g["n"] / g.groupby(["pitcher", "count_state"])["n"].transform("sum")
    g["season"] = season
    return g


def pitcher_splits(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]]
    t = _rates(_sums(reg, ["pitcher", "stand"]), "pitcher")
    t["season"] = season
    return t[["pitcher", "stand", "season"] + RATE_COLS]


def batter_splits(df: pd.DataFrame, season: int) -> dict[str, pd.DataFrame]:
    reg = df[~df["is_post"]].copy()
    fb = reg["pitch_group"] == "패스트볼"
    reg["velo_band"] = pd.cut(reg["release_speed"].where(fb), [0, 92, 95, 97, 120],
                              labels=["~92", "92~95", "95~97", "97+"])
    out = {}
    for name, key in (("vs_hand", "p_throws"), ("vs_pitch", "pitch_type"), ("vs_group", "pitch_group"),
                      ("vs_zone", "zone"), ("vs_count", "count_state"), ("vs_velo", "velo_band")):
        sub = reg.dropna(subset=[key]) if key == "velo_band" else reg
        t = _rates(_sums(sub, ["batter", key]), "batter")
        t["season"] = season
        out[name] = t[["batter", key, "season"] + RATE_COLS].rename(columns={key: "key"}).astype({"key": str})
    return out


def _grid(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    bx = np.clip(((df["plate_x"] - GRID_X[0]) / (GRID_X[1] - GRID_X[0]) * GRID_N).astype(int), 0, GRID_N - 1)
    bz = np.clip(((df["plate_z"] - GRID_Z[0]) / (GRID_Z[1] - GRID_Z[0]) * GRID_N).astype(int), 0, GRID_N - 1)
    x = df[keys].copy()
    x["gx"], x["gz"] = bx.astype("int8"), bz.astype("int8")
    x["n"] = 1
    for c in ("swing", "whiff", "bip", "is_hit", "called_strike"):
        x[c] = df[c].astype("int16")
    x["xwoba_bip"] = df["estimated_woba_using_speedangle"].where(df["bip"]).fillna(0)
    g = x.groupby(keys + ["gx", "gz"]).sum().reset_index()
    for c in ("n", "swing", "whiff", "bip", "is_hit", "called_strike"):
        g[c] = g[c].astype("int32")
    g["xwoba_bip"] = g["xwoba_bip"].astype("float32")
    return g


def location_grids(df: pd.DataFrame, season: int, p_ok, b_ok) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = df[~df["is_post"]]
    pg = _grid(reg[reg["pitcher"].isin(p_ok)], ["pitcher", "pitch_type", "stand"])
    bg = _grid(reg[reg["batter"].isin(b_ok)], ["batter", "pitch_group"])
    pg["season"] = bg["season"] = season
    return pg, bg


def spray_points(df: pd.DataFrame, season: int, b_ok) -> pd.DataFrame:
    reg = df[~df["is_post"] & df["bip"] & df["hc_x"].notna() & df["batter"].isin(b_ok)]
    ev = reg["events"].where(reg["is_hit"], "out")
    t = pd.DataFrame({"batter": reg["batter"].values, "x": (reg["hc_x"] - 125.42).astype("float16").values,
                      "y": (198.27 - reg["hc_y"]).astype("float16").values, "result": ev.astype("category").values,
                      "ev": reg["launch_speed"].astype("float16").values, "pitch_group": reg["pitch_group"].values})
    t["season"] = season
    return t


def transitions(df: pd.DataFrame, season: int) -> pd.DataFrame:
    reg = df[~df["is_post"]].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    grp = reg.groupby(["game_pk", "at_bat_number"])
    nxt = grp["pitch_type"].shift(-1)
    t = pd.DataFrame({"pitcher": reg["pitcher"], "from": reg["pitch_type"], "to": nxt,
                      # 두 번째 공(도착 구종)의 결과: 간선별 위력 계산용
                      "swing": grp["swing"].shift(-1), "whiff": grp["whiff"].shift(-1), "csw": grp["csw"].shift(-1),
                      "two_strike": grp["strikes"].shift(-1).eq(2)}).dropna(subset=["to"])
    for c in ("swing", "whiff", "csw"):
        t[c] = t[c].astype(bool).astype(int)
    t["whiff_2s"] = t["whiff"] * t["two_strike"]
    t["swing_2s"] = t["swing"] * t["two_strike"]
    g = t.groupby(["pitcher", "from", "to"]).agg(n=("swing", "size"), swing=("swing", "sum"), whiff=("whiff", "sum"),
                                                csw=("csw", "sum"), n_2s=("two_strike", "sum"), swing_2s=("swing_2s", "sum"),
                                                whiff_2s=("whiff_2s", "sum")).reset_index()
    g["season"] = season
    return g


def monthly(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = df[~df["is_post"]]
    fbp = reg[reg["pitch_group"] == "패스트볼"]
    pm = fbp.groupby(["pitcher", "month"]).agg(fb_velo=("release_speed", "mean"), n=("release_speed", "size")).reset_index()
    pa = reg[reg["pa_end"]]
    bm = pa.groupby(["batter", "month"]).agg(xwoba=("xwoba_pa", "mean"), PA=("xwoba_pa", "size")).reset_index()
    ev = reg[reg["bip"]].groupby(["batter", "month"])["launch_speed"].mean().rename("ev")
    bm = bm.merge(ev, on=["batter", "month"], how="left")
    pm["season"] = bm["season"] = season
    return pm, bm


def game_velo(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """경기별 패스트볼 평균 구속 (변화점 탐지용)."""
    reg = df[~df["is_post"] & (df["pitch_group"] == "패스트볼")]
    g = reg.groupby(["pitcher", "game_date"]).agg(fb_velo=("release_speed", "mean"),
                                                   n=("release_speed", "size")).reset_index()
    g = g[g["n"] >= 5]
    g["season"] = season
    return g


def postseason(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = df[df["is_post"]]
    if post.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = _rates(_sums(post, ["pitcher"]), "pitcher")[["pitcher"] + RATE_COLS]
    b = _rates(_sums(post, ["batter"]), "batter")[["batter"] + RATE_COLS]
    p["season"] = b["season"] = season
    return p, b


HIGHER_BETTER_P = {"K%": 1, "BB%": -1, "K-BB%": 1, "Whiff%": 1, "CSW%": 1, "Chase%": 1, "xwOBA": -1, "xBA": -1,
                   "HardHit%": -1, "Barrel%": -1, "평균 타구속도": -1, "GB%": 1, "패스트볼 평균구속": 1, "RV/100": 1,
                   "ERA": -1, "FIP": -1}
HIGHER_BETTER_B = {"xwOBA": 1, "wOBA": 1, "xBA": 1, "xSLG": 1, "평균 타구속도": 1, "EV90": 1, "최고 타구속도": 1,
                   "HardHit%": 1, "Barrel%": 1, "SweetSpot%": 1, "K%": -1, "BB%": 1, "Chase%": -1, "Whiff%": -1,
                   "Z-Contact%": 1, "배트 스피드": 1, "RV/100": 1}


def add_percentiles(t: pd.DataFrame, spec: dict) -> pd.DataFrame:
    t = t.copy()
    for season, idx in t.groupby("season").groups.items():
        q = t.loc[idx, "qualified"]
        for m, d in spec.items():
            if m not in t:
                continue
            ref = t.loc[idx][q][m].dropna()
            if len(ref) < 5:
                continue
            r = np.sort(ref.values)
            vals = t.loc[idx, m].values.astype(float)
            lo = np.searchsorted(r, vals, "left")
            hi = np.searchsorted(r, vals, "right")
            pct = (lo + hi) / 2 / len(r) * 100
            pct = np.where(np.isnan(vals), np.nan, pct)
            t.loc[idx, f"pct_{m}"] = pct if d > 0 else 100 - pct
    return t
