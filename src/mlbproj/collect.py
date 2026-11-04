"""1단계: 데이터 수집.

- Statcast 투구 단위 데이터 (Baseball Savant, pybaseball 사용, 실패 시 직접 CSV 요청)
- MLB Stats API 시즌 성적 (ERA, 이닝, 홈런, 도루 등) + 선수 인적사항(생일, 투타)
- FanGraphs WAR (선택, 실패해도 파이프라인은 계속 진행)

월 단위로 나눠 저장하므로 중간에 끊겨도 다시 실행하면 이어서 받습니다.
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from .config import EXTERNAL, KEEP_COLS, RAW, SEASONS

SAVANT_CSV = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7CPO%7C"
    "&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfSea={season}%7C&hfSit=&player_type=pitcher"
    "&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={d}&game_date_lt={d}"
    "&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0"
    "&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
    "&min_pas=0&type=details"
)
STATSAPI = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "mlb-statcast-project (data analysis research)"}


# ---------------------------------------------------------------- Statcast
def _month_ranges(season: int):
    """3월 15일 ~ 11월 10일을 월 단위 구간으로 (정규시즌+포스트시즌 포함)."""
    start, end = date(season, 3, 15), date(season, 11, 10)
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP_COLS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def _fetch_day_direct(season: int, d: date) -> pd.DataFrame:
    url = SAVANT_CSV.format(season=season, d=d.isoformat())
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            if not r.text.strip() or r.text.lstrip().startswith("<"):
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"    재시도 {attempt + 1}/4 ({d}): {e}")
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def _fetch_range(season: int, a: date, b: date) -> pd.DataFrame:
    try:
        from pybaseball import cache, statcast

        cache.enable()
        df = statcast(a.isoformat(), b.isoformat(), verbose=False)
        if df is not None and len(df):
            return df
    except Exception as e:  # noqa: BLE001
        print(f"  pybaseball 실패 → 직접 요청으로 전환: {e}")
    frames = []
    d = a
    while d <= b:
        frames.append(_fetch_day_direct(season, d))
        d += timedelta(days=1)
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_statcast(season: int, force: bool = False) -> pd.DataFrame:
    out = RAW / f"statcast_{season}.parquet"
    if out.exists() and not force:
        print(f"[statcast {season}] 이미 있음 → 건너뜀")
        return pd.read_parquet(out)
    month_dir = RAW / str(season)
    month_dir.mkdir(exist_ok=True)
    for a, b in _month_ranges(season):
        mf = month_dir / f"{a:%Y-%m}.parquet"
        if mf.exists() and not force:
            continue
        print(f"[statcast {season}] {a} ~ {b} 다운로드 중...")
        df = _fetch_range(season, a, b)
        if len(df):
            _trim(df).to_parquet(mf, index=False)
            print(f"    {len(df):,}구 저장")
        else:
            pd.DataFrame(columns=KEEP_COLS).to_parquet(mf, index=False)
    parts = [pd.read_parquet(p) for p in sorted(month_dir.glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    full.to_parquet(out, index=False)
    print(f"[statcast {season}] 총 {len(full):,}구 → {out.name}")
    return full


# ---------------------------------------------------------------- MLB Stats API
def _get_json(url: str, params: dict) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  재시도 {attempt + 1}/4: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {url}")


def _dedupe_traded(df: pd.DataFrame, games_col: str) -> pd.DataFrame:
    """트레이드된 선수가 팀별 행 + 합계 행으로 나올 수 있어 정리."""
    keep = []
    for pid, g in df.groupby("player_id"):
        if len(g) == 1:
            keep.append(g)
            continue
        top = g.loc[g[games_col].idxmax()]
        rest = g.drop(top.name)
        if rest[games_col].sum() == top[games_col]:
            keep.append(g.loc[[top.name]])            # 합계 행이 이미 있음
        else:                                          # 팀별 행만 있음 → 합산
            num = g.select_dtypes("number").drop(columns=["player_id"], errors="ignore").sum()
            row = g.iloc[[-1]].copy()
            for c in num.index:
                row[c] = num[c]
            if "outs" in num.index:   # 이닝 문자열("50.1")은 합산 불가 → 아웃 수로 다시 계산
                o = int(num["outs"])
                row["inningsPitched"] = f"{o // 3}.{o % 3}"
            row["team"] = "/".join(g["team"].astype(str))
            keep.append(row)
    return pd.concat(keep, ignore_index=True)


def collect_season_stats(season: int, group: str) -> pd.DataFrame:
    """group: 'pitching' 또는 'hitting'."""
    out = RAW / f"mlbapi_{group}_{season}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows, offset = [], 0
    while True:
        js = _get_json(f"{STATSAPI}/stats", {
            "stats": "season", "group": group, "season": season, "sportId": 1,
            "playerPool": "ALL", "gameType": "R", "limit": 1000, "offset": offset,
        })
        splits = js.get("stats", [{}])[0].get("splits", [])
        for s in splits:
            r = {"player_id": s["player"]["id"], "name": s["player"]["fullName"],
                 "team": s.get("team", {}).get("abbreviation", s.get("team", {}).get("name", ""))}
            r.update(s["stat"])
            rows.append(r)
        if len(splits) < 1000:
            break
        offset += 1000
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("name", "team", "player_id"):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= df[c].notna().sum() * 0.9:   # 대부분 숫자면 숫자로 ("123.1" 이닝 표기는 문자열 유지)
                df[c] = conv if c != "inningsPitched" else df[c]
    df = _dedupe_traded(df, "gamesPlayed" if "gamesPlayed" in df else df.columns[3])
    df["season"] = season
    df.to_parquet(out, index=False)
    print(f"[mlbapi {group} {season}] {len(df):,}명")
    return df


def collect_people(ids: list[int]) -> pd.DataFrame:
    out = RAW / "people.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["player_id"])
    need = sorted(set(int(i) for i in ids) - set(have["player_id"].astype(int)))
    rows = []
    for i in range(0, len(need), 150):
        js = _get_json(f"{STATSAPI}/people", {"personIds": ",".join(map(str, need[i:i + 150]))})
        for p in js.get("people", []):
            rows.append({
                "player_id": p["id"], "full_name": p.get("fullName"),
                "birth_date": p.get("birthDate"), "position": p.get("primaryPosition", {}).get("abbreviation"),
                "bats": p.get("batSide", {}).get("code"), "throws": p.get("pitchHand", {}).get("code"),
                "country": p.get("birthCountry"),
            })
    df = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("player_id")
    df.to_parquet(out, index=False)
    print(f"[people] {len(df):,}명")
    return df


# ---------------------------------------------------------------- FanGraphs (선택)
FG_API = ("https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats={stats}&lg=all&qual=0"
          "&season={s}&season1={s}&startdate=&enddate=&month=0&hand=&team=0&pageitems=2000000&pagenum=1"
          "&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR")
FG_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/130.0 Safari/537.36", "Accept": "application/json",
              "Referer": "https://www.fangraphs.com/leaders/major-league"}


def _fg_api(season: int, kind: str) -> pd.DataFrame:
    """FanGraphs 리더보드가 쓰는 JSON API. 결과에 MLBAM ID(xMLBAMID)가 들어 있어 바로 매칭 가능."""
    url = FG_API.format(stats="pit" if kind == "pitching" else "bat", s=season)
    r = requests.get(url, headers=FG_HEADERS, timeout=120)
    r.raise_for_status()
    rows = r.json().get("data", [])
    df = pd.DataFrame(rows)
    if "xMLBAMID" in df:
        df["MLBAMID"] = pd.to_numeric(df["xMLBAMID"], errors="coerce")
    return df


def collect_fangraphs(season: int) -> None:
    """FanGraphs fWAR·wRC+ 등 (선택). JSON API → pybaseball 순서로 시도하고, 둘 다 막히면 건너뜀.
    막혔을 때는 리더보드 CSV를 data/external 에 직접 넣으면 됨."""
    for kind in ("pitching", "batting"):
        out = EXTERNAL / f"fangraphs_{kind}_{season}.csv"
        if out.exists() and out.stat().st_size > 100:
            continue
        df = pd.DataFrame()
        try:
            df = _fg_api(season, kind)
        except Exception as e:  # noqa: BLE001
            print(f"[fangraphs {kind} {season}] API 실패: {e}")
        if df.empty or "MLBAMID" not in df:
            try:
                import pybaseball as pb

                fn = pb.pitching_stats if kind == "pitching" else pb.batting_stats
                df = fn(season, qual=0)
                ids = pb.playerid_reverse_lookup(df["IDfg"].tolist(), key_type="fangraphs")
                df = df.merge(ids[["key_fangraphs", "key_mlbam"]], left_on="IDfg",
                              right_on="key_fangraphs", how="left").rename(columns={"key_mlbam": "MLBAMID"})
            except Exception as e:  # noqa: BLE001
                print(f"[fangraphs {kind} {season}] pybaseball 실패: {e}")
                df = pd.DataFrame()
        if len(df) and "MLBAMID" in df:
            df.to_csv(out, index=False)
            print(f"[fangraphs {kind} {season}] {len(df):,}명 (WAR {'있음' if 'WAR' in df else '없음'})")
        else:
            print(f"[fangraphs {kind} {season}] 받지 못함 → fWAR 없이 진행 (wRC+는 자체 계산)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="MLB 데이터 수집")
    ap.add_argument("--seasons", default=f"{SEASONS[0]}-{SEASONS[-1]}", help="예: 2026 또는 2021-2026")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-fangraphs", action="store_true")
    a = ap.parse_args(argv)
    if "-" in a.seasons:
        s0, s1 = map(int, a.seasons.split("-"))
        seasons = list(range(s0, s1 + 1))
    else:
        seasons = [int(x) for x in a.seasons.split(",")]
    ids = set()
    for s in seasons:
        sc = collect_statcast(s, force=a.force)
        ids |= set(sc["pitcher"].dropna().astype(int)) | set(sc["batter"].dropna().astype(int))
        collect_season_stats(s, "pitching")
        collect_season_stats(s, "hitting")
        if not a.skip_fangraphs:
            collect_fangraphs(s)
    collect_people(sorted(ids))


if __name__ == "__main__":
    main()
