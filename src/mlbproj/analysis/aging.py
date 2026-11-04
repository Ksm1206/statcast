"""8단계: 에이징 커브(혼합효과모형 + 델타 방법)와 구속 변화점 탐지."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .common import load, save, save_json

AGES = np.arange(21, 41)


def _age(t: pd.DataFrame) -> pd.Series:
    a = t["age"] if "age" in t else pd.Series(np.nan, index=t.index)
    if "birth_date" in t:
        bd = pd.to_datetime(t["birth_date"], errors="coerce")
        alt = t["season"] - bd.dt.year - (bd.dt.month > 6).astype(int)
        a = a.fillna(alt)
    return a


def curve(t: pd.DataFrame, key: str, y: str, w: str):
    d = t.dropna(subset=[y, "age"]).copy()
    d = d[(d["age"] >= 20) & (d["age"] <= 42)]
    if d[key].nunique() < 20:
        return None, None
    d["ac"] = d["age"] - 28
    d["yv"] = d[y]
    res = {"n_players": int(d[key].nunique()), "n_obs": len(d)}
    try:
        fit = smf.mixedlm("yv ~ ac + I(ac**2)", d, groups=d[key]).fit(reml=True, method="lbfgs")
        b0, b1, b2 = fit.fe_params["Intercept"], fit.fe_params["ac"], fit.fe_params["I(ac ** 2)"]
        res.update({"b1": b1, "b2": b2, "p_b2": float(fit.pvalues["I(ac ** 2)"]),
                    "peak_age": float(28 - b1 / (2 * b2)) if b2 < 0 else None,
                    "random_sd": float(np.sqrt(fit.cov_re.iloc[0, 0]))})
        mixed = b0 + b1 * (AGES - 28) + b2 * (AGES - 28) ** 2
    except Exception as e:  # noqa: BLE001
        print(f"  MixedLM 실패: {e}")
        mixed = np.full(len(AGES), np.nan)
    # 델타 방법: 같은 선수의 연속 시즌 변화량을 나이별로 (조화평균 가중)
    a = d[[key, "season", "age", y, w]]
    b = a.assign(season=a["season"] - 1)
    j = a.merge(b, on=[key, "season"], suffixes=("", "_n"))
    j["delta"] = j[f"{y}_n"] - j[y]
    j["hw"] = 2 / (1 / j[w].clip(1) + 1 / j[f"{w}_n"].clip(1))
    dl = j.groupby(j["age_n"].round()).apply(lambda g: np.average(g["delta"], weights=g["hw"]) if len(g) >= 5 else np.nan,
                                             include_groups=False)
    dl = dl.reindex(AGES).fillna(0).cumsum()
    base = mixed[AGES == 28][0] if not np.isnan(mixed).all() else d[y].mean()
    dl = dl - dl.loc[28] + base
    return pd.DataFrame({"age": AGES, "혼합효과모형": mixed, "델타 방법": dl.values}), res


def changepoints(gv: pd.DataFrame, min_seg: int = 5) -> pd.DataFrame:
    rows = []
    for (p, s), g in gv.sort_values("game_date").groupby(["pitcher", "season"]):
        x = g["fb_velo"].values
        n = len(x)
        if n < 2 * min_seg + 2:
            continue
        best = (0, None)
        for k in range(min_seg, n - min_seg):
            a, b = x[:k], x[k:]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            t = (b.mean() - a.mean()) / se if se > 0 else 0
            if abs(t) > abs(best[0]):
                best = (t, k)
        t, k = best
        if k is None:
            continue
        rows.append((p, s, n, g["game_date"].iloc[k], x[:k].mean(), x[k:].mean(), x[k:].mean() - x[:k].mean(), t))
    out = pd.DataFrame(rows, columns=["pitcher", "season", "경기 수", "변화 시점", "이전 평균", "이후 평균", "변화량", "t"])
    out["구속 급락 경보"] = (out["변화량"] <= -1.0) & (out["t"] <= -3)
    return out


def main():
    meta = {}
    ps, bs = load("pitcher_season"), load("batter_season")
    ps["age"], bs["age"] = _age(ps), _age(bs)
    c1, m1 = curve(ps[ps["pitches"] >= 300], "pitcher", "패스트볼 평균구속", "pitches")
    c2, m2 = curve(bs[bs["PA"] >= 150], "batter", "xwOBA", "PA")
    if c1 is not None:
        save("aging_pitcher_velo", c1)
        meta["pitcher_velo"] = m1
    if c2 is not None:
        save("aging_batter_xwoba", c2)
        meta["batter_xwoba"] = m2
    cp = changepoints(load("pitcher_game_velo"))
    save("velo_changepoints", cp)
    meta["velo_alerts"] = int(cp["구속 급락 경보"].sum())
    save_json("aging_meta", meta)
    print(f"[aging] 구속 급락 경보 {meta['velo_alerts']}건")
