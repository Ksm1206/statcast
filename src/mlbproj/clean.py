"""2단계: 투구 데이터 정리와 파생 변수 생성."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PITCH_GROUP, RAW

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
         "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
CALLED_STRIKE = {"called_strike"}
HIT = {"single", "double", "triple", "home_run"}
AB_EVENTS = HIT | {"field_out", "strikeout", "grounded_into_double_play", "force_out", "double_play",
                   "field_error", "fielders_choice", "fielders_choice_out", "strikeout_double_play",
                   "triple_play"}
TB = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
POST_TYPES = {"F", "D", "L", "W"}


def outcome_class(ev: pd.Series) -> pd.Series:
    """타석 결과 6분류: K / BB(볼넷+사구) / 1B / XBH(2루타·3루타) / HR / OUT."""
    out = pd.Series("OUT", index=ev.index, dtype="object")
    out[ev.isin(["strikeout", "strikeout_double_play"])] = "K"
    out[ev.isin(["walk", "hit_by_pitch", "intent_walk"])] = "BB"
    out[ev == "single"] = "1B"
    out[ev.isin(["double", "triple"])] = "XBH"
    out[ev == "home_run"] = "HR"
    out[ev.isna()] = None
    return out


STR_COLS = {"game_date", "game_type", "player_name", "stand", "p_throws", "home_team", "away_team", "inning_topbot",
            "pitch_type", "pitch_name", "type", "description", "events", "bb_type"}


def _to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """pybaseball은 Int64/Float64 같은 nullable 타입을 돌려주는데, numpy 연산에서 object로 바뀌어 오류를 냄.
    숫자 컬럼은 전부 float32(결측=NaN), 문자 컬럼은 일반 object 문자열로 통일."""
    for c in df.columns:
        if c in STR_COLS:
            if c != "game_date":
                df[c] = df[c].astype(object).where(df[c].notna(), None)
            continue
        col = df[c]
        if c in ("pitcher", "batter", "game_pk", "at_bat_number", "pitch_number"):
            df[c] = pd.to_numeric(col.astype(object), errors="coerce").fillna(-1).astype("int64")
        elif col.dtype == object or pd.api.types.is_extension_array_dtype(col.dtype) or col.dtype == bool:
            df[c] = pd.to_numeric(col.astype(object), errors="coerce").astype("float32")
        elif col.dtype != "float32":
            df[c] = col.astype("float32")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def batter_zone_table(seasons: list[int]) -> pd.DataFrame:
    """타자별 스트라이크존 상·하단을 전 시즌 중앙값으로 고정.
    Statcast의 sz_top/sz_bot 산출 방식이 시즌마다 달라질 수 있어(특히 2025년부터 ABS 기준 존),
    그대로 쓰면 '존 안/밖' 판정이 데이터 정의 변화 때문에 바뀌는 착시가 생김 → 시즌 간 같은 존으로 비교."""
    parts = []
    for s in seasons:
        d = pd.read_parquet(RAW / f"statcast_{s}.parquet", columns=["batter", "sz_top", "sz_bot"])
        parts.append(_to_numpy_dtypes(d.assign(game_date="2000-01-01")).drop(columns="game_date"))
    z = pd.concat(parts).groupby("batter")[["sz_top", "sz_bot"]].median()
    return z


def load_and_clean(season: int, zone: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"statcast_{season}.parquet")
    df = _to_numpy_dtypes(df)
    if zone is not None and len(zone):
        df["sz_top_raw"], df["sz_bot_raw"] = df["sz_top"], df["sz_bot"]
        df["sz_top"] = df["batter"].map(zone["sz_top"]).fillna(df["sz_top"]).astype("float32")
        df["sz_bot"] = df["batter"].map(zone["sz_bot"]).fillna(df["sz_bot"]).astype("float32")
    df = df[df["pitch_type"].notna() & df["plate_x"].notna()].copy()
    df["is_post"] = df["game_type"].isin(POST_TYPES)
    df = df[df["game_type"].isin({"R"} | POST_TYPES)]
    df["season"] = season
    df["pitch_type"] = df["pitch_type"].replace({"FA": "FF", "FO": "FS", "CS": "CU"})
    df["pitch_group"] = df["pitch_type"].map(PITCH_GROUP).fillna("기타")

    d = df["description"]
    df["swing"] = d.isin(SWING)
    df["whiff"] = d.isin(WHIFF)
    df["called_strike"] = d.isin(CALLED_STRIKE)
    df["csw"] = df["whiff"] | df["called_strike"]
    df["take"] = ~df["swing"]
    # 존 판정: 선수별 존 상·하단 사용
    half_plate = 0.83
    df["in_zone"] = (df["plate_x"].abs() <= half_plate) & (df["plate_z"] >= df["sz_bot"]) & (df["plate_z"] <= df["sz_top"])
    df["chase"] = df["swing"] & ~df["in_zone"]
    df["zone_swing"] = df["swing"] & df["in_zone"]
    df["contact"] = df["swing"] & ~df["whiff"]
    # 경계(섀도) 존: 존 가장자리 ±약 1.5인치(0.125ft) 안팎
    edge_x = (df["plate_x"].abs() - half_plate).abs() <= 0.125
    edge_z = ((df["plate_z"] - df["sz_bot"]).abs() <= 0.125) | ((df["plate_z"] - df["sz_top"]).abs() <= 0.125)
    inside_wide = (df["plate_x"].abs() <= half_plate + 0.125) & (df["plate_z"] >= df["sz_bot"] - 0.125) & (df["plate_z"] <= df["sz_top"] + 0.125)
    df["edge"] = inside_wide & (edge_x | edge_z)
    # 좌·우 비교용 정규화: 좌투수의 수평 값 반전 (팔 쪽 = +)
    sgn = np.where(df["p_throws"] == "L", -1.0, 1.0)
    df["hb_in"] = (df["pfx_x"] * 12 * -sgn).astype("float32")   # 팔 쪽 수평 무브먼트(인치)
    df["ivb_in"] = (df["pfx_z"] * 12).astype("float32")          # 수직 무브먼트(인치)
    df["rel_x_norm"] = (df["release_pos_x"] * -sgn).astype("float32")
    # 타구
    bip = d.eq("hit_into_play") & df["launch_speed"].notna()
    df["bip"] = bip
    df["hard_hit"] = bip & (df["launch_speed"] >= 95)
    df["barrel"] = bip & (df["launch_speed_angle"] == 6)
    df["sweet_spot"] = bip & df["launch_angle"].between(8, 32)
    # 타구 방향 (스프레이 각도, 타자 기준 당겨치기 +)
    ang = np.degrees(np.arctan2(df["hc_x"] - 125.42, 198.27 - df["hc_y"]))
    df["spray_angle"] = np.where(df["stand"] == "L", -ang, ang).astype("float32")  # 우타 기준 음수=당겨침
    df["direction"] = pd.cut(df["spray_angle"], [-90, -15, 15, 90], labels=["당겨치기", "가운데", "밀어치기"])
    # 볼카운트
    df["count"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["count_state"] = np.select(
        [df["count"].eq("0-0"), df["strikes"].eq(2) & (df["balls"] < 3), df["balls"] > df["strikes"]],
        ["초구", "2스트라이크", "타자 유리"], default="동등/투수 유리")
    df.loc[df["count"].eq("3-2"), "count_state"] = "풀카운트"
    # 타석 결과
    ev = df["events"]
    df["pa_end"] = ev.notna() & (df["woba_denom"].fillna(0) > 0) | ev.isin(["strikeout", "walk"])
    df["outcome"] = outcome_class(ev.where(df["pa_end"]))
    df["is_hit"] = ev.isin(HIT)
    df["is_ab"] = ev.isin(AB_EVENTS)
    df["tb"] = ev.map(TB).fillna(0).astype("int8")
    df["xwoba_pa"] = np.where(df["pa_end"],
                              np.where(df["bip"], df["estimated_woba_using_speedangle"], df["woba_value"]), np.nan)
    df["month"] = df["game_date"].dt.month.astype("int8")
    # 야수 등판 제거: 시즌 패스트볼 평균 구속 < 84mph 이면서 300구 미만
    fb = df[df["pitch_group"] == "패스트볼"].groupby("pitcher")["release_speed"].mean()
    n = df.groupby("pitcher").size()
    position_players = fb[(fb < 84) & (n.reindex(fb.index) < 300)].index
    df = df[~df["pitcher"].isin(position_players)]
    return df.reset_index(drop=True)
