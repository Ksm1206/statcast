"""7단계: ABS 챌린지(2026) 도입 전후 비교.

- 단절 시계열(ITS): 월별 지표 ~ ABS 더미 + 추세 + 월 고정효과, Newey-West(HAC) 표준오차
- 판정 지도 비교: 격자 칸별 콜 스트라이크 비율 (도입 전 평균 vs 2026), 두 비율 z검정 + FDR
- 투수 이질성(이중차분 형태): 2025년 '경계 콜 의존도'가 높던 투수일수록 2026년 볼넷이 늘었나?
  위약(placebo) 검정: 같은 분석을 2024→2025(도입 전)에 적용해 사전 추세가 없는지 확인
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..config import ABS_SEASON
from .common import bh_fdr, load, save, save_json

METRICS = {"BB%": ("bb", "pa"), "K%": ("k", "pa"), "콜 스트라이크율(지켜본 공)": ("cs", "taken"),
           "경계 콜 스트라이크율": ("edge_cs", "edge_taken"), "존 밖 콜 스트라이크율": ("oz_cs", "oz_taken"),
           "존 안 콜 스트라이크율": ("iz_cs", "iz_taken"), "경계 투구 비율": ("edge", "pitches")}


def main():
    m = load("abs_monthly")
    if m.empty or ABS_SEASON not in set(m["season"]):
        print("[abs] ABS 시즌 데이터 없음 → 건너뜀")
        return
    for k, (a, b) in METRICS.items():
        m[k] = m[a] / m[b].replace(0, np.nan)
    m = m[m["pitches"] > 2000].sort_values(["season", "month"]).reset_index(drop=True)
    seas = m.groupby("season").apply(lambda d: pd.Series({k: d[a].sum() / d[b].sum() for k, (a, b) in METRICS.items()}),
                                     include_groups=False).reset_index()
    save("abs_season", seas)
    save("abs_monthly_rates", m[["season", "month"] + list(METRICS)])
    # ITS
    m["post"] = (m["season"] >= ABS_SEASON).astype(int)
    m["t"] = m["season"] - m["season"].min()
    X = pd.get_dummies(m["month"].astype(int), prefix="m", drop_first=True, dtype=float)
    X["post"], X["t"] = m["post"], m["t"]
    X = sm.add_constant(X)
    its = []
    for k in METRICS:
        y = m[k]
        ok = y.notna()
        fit = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        fit0 = sm.OLS(y[ok], X[ok].drop(columns="t")).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        prev = m["season"] == ABS_SEASON - 1
        its.append({"지표": k, "ABS 효과(%p)": fit.params["post"] * 100, "표준오차": fit.bse["post"] * 100,
                    "p": fit.pvalues["post"], "추세 없는 모형(%p)": fit0.params["post"] * 100, "추세 없는 p": fit0.pvalues["post"],
                    "도입 전 평균": y[m["post"] == 0].mean() * 100, f"{ABS_SEASON - 1}": y[prev].mean() * 100,
                    "2026": y[m["post"] == 1].mean() * 100})
    its = pd.DataFrame(its)
    its["q_FDR"] = bh_fdr(its["p"].values)
    save("abs_its", its)
    print(its.round(3).to_string(index=False))
    # 판정 지도
    g = load("abs_grid")
    pre = g[g["season"] < ABS_SEASON].groupby(["gx", "gz"])[["n", "cs"]].sum()
    post = g[g["season"] == ABS_SEASON].groupby(["gx", "gz"])[["n", "cs"]].sum()
    j = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner").reset_index()
    j = j[(j["n_pre"] >= 30) & (j["n_post"] >= 30)]
    p1, p2 = j["cs_pre"] / j["n_pre"], j["cs_post"] / j["n_post"]
    pp = (j["cs_pre"] + j["cs_post"]) / (j["n_pre"] + j["n_post"])
    se = np.sqrt(pp * (1 - pp) * (1 / j["n_pre"] + 1 / j["n_post"])).replace(0, np.nan)
    j["pre"], j["post"], j["diff"] = p1, p2, p2 - p1
    j["p"] = 2 * stats.norm.sf(((p2 - p1) / se).abs())
    j["q_FDR"] = bh_fdr(j["p"].values)
    save("abs_grid_diff", j)
    # 투수 이질성 + 위약 검정
    pe = load("abs_pitcher_edge")
    pe["edge_dep"] = pe["edge_called"] / pe["called"].replace(0, np.nan)
    pe["BB%"] = pe["bb"] / pe["pa"].replace(0, np.nan)
    pe["CSW%"] = pe["csw"] / pe["pitches"]

    def did(s0, s1, label):
        a = pe[(pe["season"] == s0) & (pe["pitches"] >= 300)].set_index("pitcher")
        b = pe[(pe["season"] == s1) & (pe["pitches"] >= 300)].set_index("pitcher")
        d = a[["edge_dep", "BB%", "CSW%"]].join(b[["BB%", "CSW%"]], rsuffix="_next", how="inner").dropna()
        if len(d) < 20:
            return None, None
        d["ΔBB%"], d["ΔCSW%"] = d["BB%_next"] - d["BB%"], d["CSW%_next"] - d["CSW%"]
        d["의존도 그룹"] = pd.qcut(d["edge_dep"], 4, labels=["하위 25%", "25~50%", "50~75%", "상위 25%"])
        out = {"구간": label, "n": len(d)}
        for y in ("ΔBB%", "ΔCSW%"):
            f = sm.OLS(d[y], sm.add_constant(d["edge_dep"])).fit(cov_type="HC1")
            out[f"{y} 기울기"], out[f"{y} p"] = f.params["edge_dep"], f.pvalues["edge_dep"]
        q = d.groupby("의존도 그룹", observed=True)[["ΔBB%", "ΔCSW%"]].mean()
        out["ΔBB% 상위-하위"] = q.loc["상위 25%", "ΔBB%"] - q.loc["하위 25%", "ΔBB%"]
        return out, d.reset_index()

    rows = []
    main_res, dd = did(ABS_SEASON - 1, ABS_SEASON, f"{ABS_SEASON - 1}→{ABS_SEASON} (ABS 도입)")
    if main_res:
        rows.append(main_res)
        save("abs_pitcher_did", dd)
    plc, _ = did(ABS_SEASON - 2, ABS_SEASON - 1, f"{ABS_SEASON - 2}→{ABS_SEASON - 1} (위약 검정)")
    if plc:
        rows.append(plc)
    if rows:
        save("abs_did_summary", pd.DataFrame(rows))
    save_json("abs_meta", {"abs_season": ABS_SEASON, "cells_sig": int((j["q_FDR"] < .05).sum()), "cells": len(j)})
