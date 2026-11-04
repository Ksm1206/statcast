"""Statcast와 같은 형식의 가짜 데이터 생성기 (파이프라인·앱 테스트용).

실제 분석 결과가 아닙니다. 컬럼 구조와 의미만 Statcast/MLB Stats API를 흉내 냅니다.
사용: MLB_RAW_DIR=/tmp/raw python tests/make_synthetic.py --seasons 2024-2026
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlbproj.config import KEEP_COLS, RAW  # noqa: E402

rng = np.random.default_rng(7)
TEAMS = ["LAD", "NYY", "TOR", "SEA", "MIL", "TB", "CLE", "PHI", "HOU", "ATL"]
FIRST = ["Alex", "Ben", "Chris", "Dan", "Eli", "Felix", "Gabe", "Hank", "Ivan", "Jose",
         "Kyle", "Luis", "Max", "Nate", "Omar", "Paul", "Ryan", "Sam", "Tyler", "Will"]
LAST = ["Smith", "Garcia", "Kim", "Lee", "Park", "Diaz", "Lopez", "Brown", "Ohtani", "Cole",
        "Ward", "Reyes", "Torres", "Hall", "King", "Young", "Cruz", "Baker", "Rivera", "Ortiz"]

ARCH = {  # 구종: (구속, 회전, pfx_x(인치, 우투 기준), pfx_z(인치))
    "FF": (95, 2350, -7, 16), "SI": (94, 2150, -15, 8), "FC": (90, 2400, 2, 9),
    "SL": (86, 2450, 5, 2), "ST": (82, 2600, 15, 1), "CU": (79, 2650, 8, -10),
    "CH": (86, 1750, -14, 6), "FS": (87, 1300, -10, 3), "KC": (82, 2500, 7, -8),
}
ARSENALS = [["FF", "SL", "CH"], ["FF", "ST", "CU", "CH"], ["SI", "SL", "CH"],
            ["FF", "FC", "CU"], ["SI", "FC", "ST", "FS"], ["FF", "SL", "CU", "CH", "SI"],
            ["FF", "FS", "SL"], ["SI", "KC", "CH"]]


def make_players(n_p=140, n_b=170):
    pit, bat = [], []
    for i in range(n_p):
        ars = ARSENALS[rng.integers(len(ARSENALS))]
        use = rng.dirichlet(np.ones(len(ars)) * 2.5)
        use[0] += 0.25
        use /= use.sum()
        pit.append(dict(
            id=600000 + i, name=f"{LAST[i % 20]}, {FIRST[(i * 7) % 20]}{i}", throws="L" if rng.random() < .28 else "R",
            arsenal=ars, usage=use, velo_off=rng.normal(0, 1.6), stuff=rng.normal(0, 1), command=rng.normal(0, 1),
            ext=rng.normal(6.4, .35), rel_z=rng.normal(5.9, .35), birth=1990 + rng.integers(0, 12),
            team=TEAMS[i % 10], decline=rng.random() < .1))
    for i in range(n_b):
        bat.append(dict(
            id=660000 + i, name=f"{FIRST[i % 20]} {LAST[(i * 3) % 20]}{i}", stand=rng.choice(["R", "L", "S"], p=[.55, .37, .08]),
            contact=rng.normal(0, 1), power=rng.normal(0, 1), eye=rng.normal(0, 1), speed=rng.normal(0, 1),
            weak=rng.choice(list(ARCH)), birth=1990 + rng.integers(0, 12), team=TEAMS[i % 10]))
    return pit, bat


def in_zone(x, z):
    return abs(x) <= 0.83 and 1.6 <= z <= 3.4


def zone_id(x, z):
    if in_zone(x, z):
        col = min(int((x + 0.83) / (1.66 / 3)), 2)
        row = min(int((3.4 - z) / (1.8 / 3)), 2)
        return 1 + row * 3 + col
    return (11 if x < 0 else 12) if z >= 2.5 else (13 if x < 0 else 14)


def xba_xwoba(ev, la):
    sweet = np.exp(-((la - 18) / 14) ** 2)
    xba = np.clip(0.12 + 0.75 * sweet * (1 / (1 + np.exp(-(ev - 92) / 4))) + 0.1 * (la < 10) * (ev > 95), 0.01, 0.95)
    hr = (ev > 99) & (la > 22) & (la < 38)
    xwoba = np.clip(xba * 1.15 + 0.9 * hr * (ev - 99) / 10, 0.01, 1.9)
    return float(xba), float(xwoba)


WOBA = {"walk": .69, "hit_by_pitch": .72, "single": .88, "double": 1.25, "triple": 1.58, "home_run": 2.03}


def simulate_season(season, pit, bat, n_pa=32000):
    rows = []
    gpk = season * 10000
    abs_on = season >= 2026
    p_w = np.array([1.5 if i < 50 else 0.6 for i in range(len(pit))])
    p_w /= p_w.sum()
    days = pd.date_range(f"{season}-03-27", f"{season}-09-28")
    post_days = pd.date_range(f"{season}-10-01", f"{season}-10-30")
    for pa in range(n_pa):
        post = pa >= n_pa - 600
        if pa % 70 == 0:
            gpk += 1
            gdate = post_days[rng.integers(len(post_days))] if post else days[rng.integers(len(days))]
            ab_no = 0
            home, away = rng.choice(TEAMS, 2, replace=False)
        ab_no += 1
        P = pit[rng.choice(len(pit), p=p_w)]
        B = bat[rng.integers(len(bat))]
        age_p = season - P["birth"]
        age_b = season - B["birth"]
        stand = B["stand"] if B["stand"] != "S" else ("L" if P["throws"] == "R" else "R")
        balls = strikes = 0
        pitch_no = 0
        month_frac = (gdate.month - 3) / 7
        vel_drop = -2.0 * month_frac if (P["decline"] and season == max(2026, season)) else 0
        aging = -0.15 * max(age_p - 29, 0)
        while True:
            pitch_no += 1
            use = P["usage"].copy()
            if strikes == 2:
                use[1:] *= 1.4
            if balls >= 2 and balls > strikes:
                use[0] *= 1.6
            use /= use.sum()
            pt = P["arsenal"][rng.choice(len(use), p=use)]
            v0, sp0, hx0, vz0 = ARCH[pt]
            sgn = -1 if P["throws"] == "L" else 1
            velo = v0 + P["velo_off"] + aging + vel_drop + rng.normal(0, .9)
            spin = sp0 + 60 * P["stuff"] + rng.normal(0, 70)
            pfx_x = sgn * (hx0 + rng.normal(0, 1.5)) / 12
            pfx_z = (vz0 + 1.2 * P["stuff"] + rng.normal(0, 1.5)) / 12
            tx = (0.45 if pt not in ("FF", "SI") else 0.1) * (1 if stand == "R" else -1) * sgn
            tz = 1.7 if pt in ("SL", "ST", "CU", "CH", "FS", "KC") else 2.8
            sd = 0.75 - 0.08 * P["command"]
            x, z = tx + rng.normal(0, sd), tz + rng.normal(0, sd)
            iz = in_zone(x, z)
            edge = abs(abs(x) - 0.83) < 0.1 or abs(z - 1.6) < 0.1 or abs(z - 3.4) < 0.1
            sw_p = (0.66 if iz else 0.28 - 0.04 * B["eye"]) + 0.08 * (strikes == 2)
            swing = rng.random() < np.clip(sw_p, .05, .95)
            ev = la = xba = xw = ls_ang = bat_speed = swing_len = np.nan
            events = bb_type = None
            desc, typ = "ball", "B"
            if swing:
                bat_speed = rng.normal(71.5 + 1.5 * B["power"], 3) if season >= 2024 else np.nan
                swing_len = rng.normal(7.3 + .3 * B["power"], .5) if season >= 2024 else np.nan
                whiff = np.clip(0.22 + 0.05 * P["stuff"] - 0.05 * B["contact"] + 0.08 * (not iz) + 0.07 * (pt == B["weak"])
                                + 0.04 * (pt in ("ST", "SL", "FS", "CU")), .03, .8)
                if rng.random() < whiff:
                    desc, typ = "swinging_strike", "S"
                elif rng.random() < 0.48:
                    desc, typ = "foul", "S"
                else:
                    desc, typ = "hit_into_play", "X"
                    ev = rng.normal(89 + 3.5 * B["power"] - 1.5 * P["stuff"] - 3 * (not iz), 12)
                    la = rng.normal(12 + 3 * B["power"] - 4 * (pt in ("SI",)), 24)
                    xba, xw = xba_xwoba(ev, la)
                    ls_ang = 6 if (ev >= 98 and 26 - (ev - 98) <= la <= 30 + (ev - 98)) else rng.integers(1, 6)
                    bb_type = "ground_ball" if la < 10 else "line_drive" if la < 25 else "fly_ball" if la < 50 else "popup"
                    r = rng.random()
                    if xw > 1.3 and r < 0.75:
                        events = "home_run"
                    elif r < xba:
                        events = rng.choice(["single", "double", "triple"], p=[.68, .29, .03])
                    else:
                        events = rng.choice(["field_out", "grounded_into_double_play", "force_out"], p=[.85, .1, .05])
            else:
                if abs_on:
                    called = iz if not edge else (rng.random() < (0.55 if iz else 0.35))
                else:
                    called = iz if not edge else (rng.random() < (0.45 if iz else 0.45))
                if called:
                    desc, typ = "called_strike", "S"
                elif rng.random() < 0.008:
                    desc, typ, events = "hit_by_pitch", "B", "hit_by_pitch"
            prev_b, prev_s = balls, strikes
            if events is None:
                if typ == "B":
                    balls += 1
                    if balls == 4:
                        events = "walk"
                elif desc == "foul":
                    strikes = min(strikes + 1, 2)
                else:
                    strikes += 1
                    if strikes == 3:
                        events = "strikeout"
            dre = {"ball": .035, "called_strike": -.045, "swinging_strike": -.05, "foul": -.02,
                   "hit_by_pitch": .3}.get(desc, 0.0)
            if events in WOBA:
                dre = (WOBA[events] - .3) * .9
            elif events in ("field_out", "force_out", "grounded_into_double_play"):
                dre = -.25
            elif events == "strikeout":
                dre = -.27
            spray = rng.normal(0, 25) + (-12 if stand == "R" else 12) * (B["power"] > 0.5)
            dist = 140 + 2.2 * max(ev - 70, 0) if typ == "X" else np.nan
            rows.append(dict(
                game_date=gdate, game_year=season, game_type=("W" if post else "R"), game_pk=gpk,
                at_bat_number=ab_no, pitch_number=pitch_no, pitcher=P["id"], batter=B["id"],
                player_name=P["name"], stand=stand, p_throws=P["throws"], home_team=home, away_team=away,
                inning=min(1 + ab_no // 8, 9), inning_topbot="Top" if ab_no % 2 else "Bot", outs_when_up=ab_no % 3,
                balls=prev_b, strikes=prev_s, on_1b=np.nan, on_2b=np.nan, on_3b=np.nan,
                pitch_type=pt, pitch_name=pt, release_speed=velo, release_spin_rate=spin, spin_axis=rng.normal(200, 30),
                pfx_x=pfx_x, pfx_z=pfx_z, plate_x=x, plate_z=z, sz_top=3.4, sz_bot=1.6, zone=zone_id(x, z),
                release_pos_x=sgn * -2.0 + rng.normal(0, .15), release_pos_z=P["rel_z"] + rng.normal(0, .1),
                release_extension=P["ext"] + rng.normal(0, .1),
                vx0=np.nan, vy0=np.nan, vz0=np.nan, ax=np.nan, ay=np.nan, az=np.nan,
                type=typ, description=desc, events=events, bb_type=bb_type, launch_speed=ev, launch_angle=la,
                launch_speed_angle=ls_ang,
                hc_x=125.42 + dist / 2.5 * np.sin(np.radians(spray)) if typ == "X" else np.nan,
                hc_y=198.27 - dist / 2.5 * np.cos(np.radians(spray)) if typ == "X" else np.nan,
                estimated_ba_using_speedangle=xba, estimated_woba_using_speedangle=xw,
                estimated_slg_using_speedangle=xw * 1.3 if typ == "X" else np.nan,
                woba_value=WOBA.get(events, 0.0) if events else np.nan,
                woba_denom=1.0 if events else np.nan, babip_value=np.nan, iso_value=np.nan,
                delta_run_exp=dre, bat_speed=bat_speed, swing_length=swing_len,
                n_thruorder_pitcher=1 + (ab_no // 9) % 3, age_pit=age_p, age_bat=age_b))
            if events:
                break
    df = pd.DataFrame(rows)[KEEP_COLS]
    return df


def season_stats(df, pit, bat, season):
    reg = df[df.game_type == "R"]
    ev = reg[reg.events.notna()]
    p = ev.groupby("pitcher").agg(
        battersFaced=("events", "size"), strikeOuts=("events", lambda s: (s == "strikeout").sum()),
        baseOnBalls=("events", lambda s: (s == "walk").sum()), hitByPitch=("events", lambda s: (s == "hit_by_pitch").sum()),
        homeRuns=("events", lambda s: (s == "home_run").sum()),
        hits=("events", lambda s: s.isin(["single", "double", "triple", "home_run"]).sum()),
        outs=("events", lambda s: s.isin(["strikeout", "field_out", "force_out"]).sum() + 2 * (s == "grounded_into_double_play").sum()),
        gamesPlayed=("game_pk", "nunique"),
    ).reset_index().rename(columns={"pitcher": "player_id"})
    p["earnedRuns"] = (0.45 * p.hits + 0.33 * (p.baseOnBalls + p.hitByPitch) + 1.0 * p.homeRuns + rng.normal(0, 3, len(p))).clip(0).round()
    p["inningsPitched"] = (p.outs // 3).astype(str) + "." + (p.outs % 3).astype(str)
    p["era"] = (27 * p.earnedRuns / p.outs.clip(1)).round(2)
    p["whip"] = (3 * (p.hits + p.baseOnBalls) / p.outs.clip(1)).round(2)
    p["wins"] = rng.integers(0, 15, len(p))
    p["losses"] = rng.integers(0, 12, len(p))
    p["saves"] = 0
    pm = {x["id"]: x for x in pit}
    p["name"] = p.player_id.map(lambda i: " ".join(pm[i]["name"].split(", ")[::-1]))
    p["team"] = p.player_id.map(lambda i: pm[i]["team"])
    p["season"] = season
    b = ev.groupby("batter").agg(
        plateAppearances=("events", "size"), strikeOuts=("events", lambda s: (s == "strikeout").sum()),
        baseOnBalls=("events", lambda s: (s == "walk").sum()), hitByPitch=("events", lambda s: (s == "hit_by_pitch").sum()),
        homeRuns=("events", lambda s: (s == "home_run").sum()), doubles=("events", lambda s: (s == "double").sum()),
        triples=("events", lambda s: (s == "triple").sum()),
        hits=("events", lambda s: s.isin(["single", "double", "triple", "home_run"]).sum()),
        gamesPlayed=("game_pk", "nunique"),
    ).reset_index().rename(columns={"batter": "player_id"})
    b["atBats"] = b.plateAppearances - b.baseOnBalls - b.hitByPitch
    b["sacFlies"] = 0
    b["avg"] = (b.hits / b.atBats.clip(1)).round(3)
    b["obp"] = ((b.hits + b.baseOnBalls + b.hitByPitch) / b.plateAppearances.clip(1)).round(3)
    tb = b.hits + b.doubles + 2 * b.triples + 3 * b.homeRuns
    b["slg"] = (tb / b.atBats.clip(1)).round(3)
    b["ops"] = b.obp + b.slg
    bm = {x["id"]: x for x in bat}
    b["stolenBases"] = b.player_id.map(lambda i: max(0, int(rng.normal(6 + 6 * bm[i]["speed"], 3))))
    b["rbi"] = (b.hits * .5 + b.homeRuns).round()
    b["runs"] = (b.hits * .45 + b.homeRuns).round()
    b["name"] = b.player_id.map(lambda i: bm[i]["name"])
    b["team"] = b.player_id.map(lambda i: bm[i]["team"])
    b["season"] = season
    return p, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024-2026")
    ap.add_argument("--pa", type=int, default=32000)
    a = ap.parse_args()
    s0, s1 = map(int, a.seasons.split("-"))
    pit, bat = make_players()
    for s in range(s0, s1 + 1):
        df = simulate_season(s, pit, bat, a.pa)
        df.to_parquet(RAW / f"statcast_{s}.parquet", index=False)
        p, b = season_stats(df, pit, bat, s)
        p.to_parquet(RAW / f"mlbapi_pitching_{s}.parquet", index=False)
        b.to_parquet(RAW / f"mlbapi_hitting_{s}.parquet", index=False)
        print(s, len(df), "pitches")
    people = [dict(player_id=x["id"], full_name=" ".join(x["name"].split(", ")[::-1]), birth_date=f"{x['birth']}-06-15",
                   position="P", bats="R", throws=x["throws"], country="USA") for x in pit]
    people += [dict(player_id=x["id"], full_name=x["name"], birth_date=f"{x['birth']}-06-15", position="OF",
                    bats=x["stand"], throws="R", country="USA") for x in bat]
    pd.DataFrame(people).to_parquet(RAW / "people.parquet", index=False)


if __name__ == "__main__":
    main()
