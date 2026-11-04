"""5단계(공간): 스트라이크존 격자의 Global Moran's I 와 LISA (로컬 모란).

격자 칸 = 공간 단위, 인접 = rook(상하좌우). 값이 비슷한 칸끼리 뭉쳐 있으면 I > 0.
LISA로 칸별 HH(핫스팟)/LL(콜드스팟)/HL/LH 유형을 순열검정으로 판정합니다.
앱에서도 같은 함수를 불러 선수별로 즉석 계산합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GRID_N
from .common import bh_fdr, load, save


def rook_weights(cells: list[tuple[int, int]]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(cells)}
    W = np.zeros((len(cells), len(cells)))
    for (x, z), i in idx.items():
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((x + dx, z + dz))
            if j is not None:
                W[i, j] = 1
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)


def morans_i(y: np.ndarray, W: np.ndarray, perms: int = 499, seed: int = 0) -> tuple[float, float]:
    y = np.asarray(y, float)
    z = y - y.mean()
    den = z @ z
    if den == 0 or len(y) < 6:
        return np.nan, np.nan
    s0 = W.sum()
    I = len(y) / s0 * (z @ W @ z) / den
    rng = np.random.default_rng(seed)
    sims = np.empty(perms)
    for k in range(perms):
        zp = rng.permutation(z)
        sims[k] = len(y) / s0 * (zp @ W @ zp) / den
    p = (1 + (np.abs(sims - sims.mean()) >= abs(I - sims.mean())).sum()) / (perms + 1)
    return float(I), float(p)


def lisa(y: np.ndarray, W: np.ndarray, perms: int = 499, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    y = np.asarray(y, float)
    z = (y - y.mean()) / (y.std() or 1)
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(seed)
    n = len(y)
    p = np.ones(n)
    nb = (W > 0).sum(axis=1)
    for i in range(n):
        if nb[i] == 0:
            continue
        others = np.delete(z, i)
        # 조건부 순열: 이웃 수만큼 다른 칸 값을 무작위로 뽑아 평균 (벡터화)
        idx = np.argsort(rng.random((perms, n - 1)), axis=1)[:, : nb[i]]
        sims = z[i] * others[idx].mean(axis=1)
        p[i] = (1 + (np.abs(sims) >= abs(Ii[i])).sum()) / (perms + 1)
    q = bh_fdr(p)
    typ = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    typ = np.where(p < alpha, typ, "유의하지 않음")
    return pd.DataFrame({"I_local": Ii, "p": p, "q": q, "유형": typ})


def grid_metric(grid: pd.DataFrame, metric: str, min_n: int = 8) -> pd.DataFrame:
    g = grid.groupby(["gx", "gz"])[["n", "swing", "whiff", "bip", "xwoba_bip", "is_hit", "called_strike"]].sum().reset_index()
    if metric == "whiff":
        g["value"], g["den"] = g["whiff"] / g["swing"].replace(0, np.nan), g["swing"]
    elif metric == "xwoba":
        g["value"], g["den"] = g["xwoba_bip"] / g["bip"].replace(0, np.nan), g["bip"]
    elif metric == "swing":
        g["value"], g["den"] = g["swing"] / g["n"].replace(0, np.nan), g["n"]
    else:
        g["value"], g["den"] = g["n"] / g["n"].sum(), g["n"]
    return g[g["den"] >= min_n].dropna(subset=["value"])


def main():
    bg = load("batter_grid")
    rows = []
    for (b, s), g in bg.groupby(["batter", "season"]):
        m = grid_metric(g, "whiff", min_n=8)
        if len(m) < 12:
            continue
        W = rook_weights(list(zip(m["gx"], m["gz"])))
        I, p = morans_i(m["value"].values, W, perms=199)
        rows.append((b, s, len(m), I, p))
    t = pd.DataFrame(rows, columns=["batter", "season", "cells", "MoranI_whiff", "p"])
    t["q_FDR"] = bh_fdr(t["p"].values)
    save("batter_moran", t)
    # 리그 전체 구종 그룹별 헛스윙 지도 Moran's I
    lg = []
    for (pg_, s), g in bg.groupby(["pitch_group", "season"]):
        m = grid_metric(g, "whiff", min_n=30)
        if len(m) < 12:
            continue
        W = rook_weights(list(zip(m["gx"], m["gz"])))
        I, p = morans_i(m["value"].values, W, perms=499)
        lg.append((pg_, s, I, p))
    save("league_moran", pd.DataFrame(lg, columns=["pitch_group", "season", "MoranI_whiff", "p"]))
    print(f"[spatial] 타자 {len(t)}명, 유의 비율 {(t['q_FDR'] < .05).mean():.0%}  (격자 {GRID_N}x{GRID_N})")
