"""볼배합 네트워크 분석.

투수마다 구종 = 노드, '같은 타석에서 A 다음 B를 던짐' = 방향 간선(가중치 = 횟수)으로 그래프를 만든다.
- PageRank: 볼배합의 허브 구종 (다른 구종이 자주 넘어오는 공)
- 조건부 엔트로피: 다음 공의 예측 어려움 (bit)
- 간선 위력: 도착 구종의 헛스윙률, 2스트라이크 헛스윙률
리그 분석: 엔트로피가 높은(예측하기 어려운) 투수일수록 삼진을 더 잡는가? 구위를 통제한 회귀로 확인.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import load, save, save_json


def pagerank(P: np.ndarray, d: float = 0.85, iters: int = 100) -> np.ndarray:
    n = len(P)
    r = np.full(n, 1 / n)
    for _ in range(iters):
        r_new = (1 - d) / n + d * r @ P
        if np.abs(r_new - r).sum() < 1e-10:
            break
        r = r_new
    return r / r.sum()


def build_graph(t: pd.DataFrame, min_share: float = 0.03):
    """한 투수·시즌의 전이 테이블 → (구종 목록, 전이확률 행렬, 출발 비율, 간선 표)"""
    out_n = t.groupby("from")["n"].sum()
    keep = out_n[out_n / out_n.sum() >= min_share].index
    t = t[t["from"].isin(keep) & t["to"].isin(keep)]
    types = list(out_n.loc[keep].sort_values(ascending=False).index)
    if len(types) < 2:
        return types, None, None, t
    M = t.pivot_table(index="from", columns="to", values="n", aggfunc="sum").reindex(index=types, columns=types).fillna(0)
    P = M.div(M.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).values
    pi = M.sum(axis=1).values / M.values.sum()
    return types, P, pi, t


def summarize(t: pd.DataFrame):
    types, P, pi, e = build_graph(t)
    if P is None:
        return None, None
    pr = pagerank(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_rows = -np.nansum(np.where(P > 0, P * np.log2(P), 0), axis=1)
    H = float((pi * H_rows).sum())
    H_max = float(np.log2(len(types)))
    nodes = pd.DataFrame({"pitch_type": types, "pagerank": pr, "출발 비율": pi, "다음 공 엔트로피": H_rows})
    e = e.assign(whiff_rate=e["whiff"] / e["swing"].replace(0, np.nan), whiff_2s_rate=e["whiff_2s"] / e["swing_2s"].replace(0, np.nan))
    best = e[e["swing"] >= 15].sort_values("whiff_rate", ascending=False).head(1)
    best2 = e[e["swing_2s"] >= 10].sort_values("whiff_2s_rate", ascending=False).head(1)
    summ = {"구종 수(네트워크)": len(types), "엔트로피": H, "엔트로피(정규화)": H / H_max if H_max > 0 else np.nan,
            "허브 구종": types[int(np.argmax(pr))], "허브 PageRank": float(pr.max()),
            "최고 위력 연결": f"{best.iloc[0]['from']}→{best.iloc[0]['to']}" if len(best) else None,
            "최고 연결 헛스윙률": float(best.iloc[0]["whiff_rate"]) if len(best) else np.nan,
            "2S 결정 연결": f"{best2.iloc[0]['from']}→{best2.iloc[0]['to']}" if len(best2) else None,
            "2S 결정 헛스윙률": float(best2.iloc[0]["whiff_2s_rate"]) if len(best2) else np.nan}
    return summ, nodes


def main():
    tr = load("pitcher_transitions")
    ps = load("pitcher_season")
    if tr.empty or "swing" not in tr:
        print("[network] 전이 데이터 없음 → 건너뜀")
        return
    q = ps.loc[ps["qualified"], ["pitcher", "season"]]
    tr = tr.merge(q, on=["pitcher", "season"])
    rows, node_rows = [], []
    for (p, s), t in tr.groupby(["pitcher", "season"]):
        summ, nodes = summarize(t)
        if summ is None:
            continue
        rows.append({"pitcher": p, "season": s, **summ})
        node_rows.append(nodes.assign(pitcher=p, season=s))
    summ = pd.DataFrame(rows)
    save("network_summary", summ)
    save("network_nodes", pd.concat(node_rows, ignore_index=True))
    # 리그 분석: 엔트로피 → K%, xwOBA (구위·구종 수 통제)
    st = load("pitcher_stuff")
    d = summ.merge(ps[["pitcher", "season", "K%", "xwOBA", "Whiff%", "pitches"]], on=["pitcher", "season"])
    if len(st):
        d = d.merge(st, on=["pitcher", "season"], how="left")
    d = d.dropna(subset=["엔트로피(정규화)", "K%", "xwOBA"])
    meta = {"n": len(d)}
    ctrl = [c for c in ("구위 지수", "구종 수(네트워크)") if c in d]
    for y in ("K%", "xwOBA"):
        X = d[["엔트로피(정규화)"] + ctrl].astype(float)
        X = (X - X.mean()) / X.std()
        ok = X.notna().all(axis=1)
        f = sm.OLS(d.loc[ok, y].astype(float), sm.add_constant(X[ok])).fit(cov_type="HC1")
        meta[f"{y}_coef_entropy"] = float(f.params["엔트로피(정규화)"])
        meta[f"{y}_p_entropy"] = float(f.pvalues["엔트로피(정규화)"])
        meta[f"{y}_r2"] = float(f.rsquared)
        meta[f"{y}_raw_corr"] = float(d["엔트로피(정규화)"].corr(d[y]))
    hub = summ["허브 구종"].value_counts(normalize=True).head(6)
    meta["hub_share"] = hub.to_dict()
    save_json("network_meta", meta)
    print(f"[network] {len(summ)}명·시즌, 엔트로피→K% 표준화 계수 {meta['K%_coef_entropy']:+.4f} (p={meta['K%_p_entropy']:.3f})")
