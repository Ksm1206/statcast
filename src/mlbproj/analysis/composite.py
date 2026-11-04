"""종합 지수: 엔트로피 가중법 vs SHAP 가중치 → Min-Max 0~100.

1) 지표 선정 + 방향 통일 (낮을수록 좋은 지표는 뒤집음)
2) 시즌별 Min-Max 정규화 (0~1)
3) 가중치
   - 엔트로피 가중법: 선수 간 차이가 큰(정보량이 많은) 지표에 큰 가중치  w_j = (1-e_j) / Σ(1-e_k)
   - SHAP 가중법: LightGBM으로 실점 가치(투수 RV/100, 타자 wOBA)를 예측하고 평균 |SHAP| 비율
4) 가중합 → 시즌별 Min-Max 0~100점
5) 검증: 같은 시즌 성적과의 상관, 다음 시즌 성적 예측력 → 더 나은 쪽을 최종 지수로
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from .common import load, save, save_json

SPEC = {
    "pitcher": {"K%": 1, "BB%": -1, "xwOBA": -1, "Whiff%": 1, "Chase%": 1, "CSW%": 1, "HardHit%": -1,
                "Barrel%": -1, "GB%": 1, "구위 지수": 1},
    "batter": {"xwOBA": 1, "Barrel%": 1, "HardHit%": 1, "BB%": 1, "K%": -1, "Chase%": -1, "Z-Contact%": 1,
               "SweetSpot%": 1, "배트 스피드": 1},
}
TARGET = {"pitcher": "RV/100", "batter": "wOBA"}
CHECK = {"pitcher": [("ERA", -1), ("FIP", -1), ("RV/100", 1)], "batter": [("wRC+", 1), ("OPS", 1), ("wOBA", 1)]}


def entropy_weights(X: pd.DataFrame) -> pd.Series:
    P = (X + 1e-9).div((X + 1e-9).sum(axis=0), axis=1)
    k = 1 / np.log(len(X))
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def shap_weights(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, verbose=-1, seed=0)
    m = lgb.train(params, lgb.Dataset(X, y), 400)
    sh = np.abs(m.predict(X, pred_contrib=True)[:, :-1]).mean(axis=0)
    return pd.Series(sh / sh.sum(), index=X.columns)


def run(kind: str):
    t = load(f"{kind}_season")
    if kind == "pitcher":
        st = load("pitcher_stuff")
        if len(st):
            t = t.merge(st, on=["pitcher", "season"], how="left")
    t = t[t["qualified"]].copy()
    spec = {m: d for m, d in SPEC[kind].items() if m in t and t[m].notna().mean() > 0.7}
    t = t.dropna(subset=list(spec) + [TARGET[kind]])
    # 방향 통일 + 시즌별 Min-Max
    N = pd.DataFrame(index=t.index)
    for m, d in spec.items():
        v = t[m] * d
        g = v.groupby(t["season"])
        N[m] = (v - g.transform("min")) / (g.transform("max") - g.transform("min")).replace(0, np.nan)
    N = N.fillna(0.5)
    w_ent = entropy_weights(N)
    w_shap = shap_weights(t[list(spec)].astype(float), t[TARGET[kind]].astype(float))

    def score(w):
        raw = N[list(spec)] @ w[list(spec)]
        g = raw.groupby(t["season"])
        return 100 * (raw - g.transform("min")) / (g.transform("max") - g.transform("min"))

    out = t[[kind, "season", "name"]].copy()
    out["엔트로피 지수"], out["SHAP 지수"] = score(w_ent), score(w_shap)
    # 검증
    val = []
    nxt = out.merge(t[[kind, "season", TARGET[kind]]].assign(season=t["season"] - 1), on=[kind, "season"], suffixes=("", "_next"))
    for col in ("엔트로피 지수", "SHAP 지수"):
        row = {"지수": col}
        for c, sign in CHECK[kind]:
            if c in t:
                row[f"같은 시즌 {c} ρ"] = sign * stats.spearmanr(out[col], t[c], nan_policy="omit").statistic
        row[f"다음 시즌 {TARGET[kind]} ρ"] = stats.spearmanr(nxt[col], nxt[TARGET[kind]], nan_policy="omit").statistic if len(nxt) > 20 else np.nan
        val.append(row)
    base = "xwOBA"
    bx = t[[kind, "season", base]].merge(t[[kind, "season", TARGET[kind]]].assign(season=t["season"] - 1), on=[kind, "season"])
    sign = -1 if kind == "pitcher" else 1
    val.append({"지수": f"비교: {base} 하나만", f"다음 시즌 {TARGET[kind]} ρ": sign * stats.spearmanr(bx[base], bx[TARGET[kind]]).statistic})
    val = pd.DataFrame(val)
    col_next = f"다음 시즌 {TARGET[kind]} ρ"
    best = val.iloc[:2].sort_values(col_next, ascending=False).iloc[0]["지수"]
    out["종합 지수"] = out[best]
    out["시즌 내 백분위"] = out.groupby("season")["종합 지수"].rank(pct=True) * 100
    W = pd.DataFrame({"지표": list(spec), "방향": [("높을수록 좋음" if d > 0 else "낮을수록 좋음") for d in spec.values()],
                      "엔트로피 가중치": w_ent[list(spec)].values, "SHAP 가중치": w_shap[list(spec)].values})
    save(f"composite_{kind}", out)
    save(f"composite_weights_{kind}", W)
    save(f"composite_validation_{kind}", val)
    agree = float(stats.spearmanr(out["엔트로피 지수"], out["SHAP 지수"]).statistic)
    return {"final": best, "n": len(out), "entropy_vs_shap_spearman": agree,
            "top_entropy_weight": W.sort_values("엔트로피 가중치").iloc[-1]["지표"],
            "top_shap_weight": W.sort_values("SHAP 가중치").iloc[-1]["지표"]}


def main():
    meta = {}
    for kind in ("pitcher", "batter"):
        meta[kind] = run(kind)
        print(f"[composite] {kind}: 최종 = {meta[kind]['final']}, 엔트로피 최대 가중 {meta[kind]['top_entropy_weight']}, "
              f"SHAP 최대 가중 {meta[kind]['top_shap_weight']}")
    save_json("composite_meta", meta)
