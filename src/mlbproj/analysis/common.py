"""분석 모듈 공통 도구: 저장/로드, VIF, FDR, Newey-West, Ljung-Box."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ..config import INTERIM, PROCESSED


def load(name: str) -> pd.DataFrame:
    for d in (PROCESSED, INTERIM):
        f = d / f"{name}.parquet"
        if f.exists():
            return pd.read_parquet(f)
    return pd.DataFrame()


def save(name: str, df: pd.DataFrame) -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    df.to_parquet(PROCESSED / f"{name}.parquet", index=False, compression="zstd")
    print(f"  → {name}.parquet  {len(df):,}행")


def save_json(name: str, obj) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (PROCESSED / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv))
    print(f"  → {name}.json")


def vif_filter(X: pd.DataFrame, thresh: float = 10.0) -> tuple[list[str], list[tuple[str, float]]]:
    """VIF가 thresh를 넘는 변수를 하나씩 제거. (남은 변수, [(제거 변수, VIF)])"""
    cols = list(X.columns)
    removed = []
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    while len(cols) > 2:
        vifs = {}
        for c in cols:
            others = [o for o in cols if o != c]
            A = np.column_stack([np.ones(len(Z)), Z[others].values])
            beta, *_ = np.linalg.lstsq(A, Z[c].values, rcond=None)
            resid = Z[c].values - A @ beta
            r2 = 1 - resid.var() / max(Z[c].values.var(), 1e-12)
            vifs[c] = 1 / max(1 - r2, 1e-9)
        worst = max(vifs, key=vifs.get)
        if vifs[worst] <= thresh:
            break
        removed.append((worst, float(vifs[worst])))
        cols.remove(worst)
    return cols, removed


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-value."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def newey_west_se(x: np.ndarray, lag: int | None = None) -> float:
    """평균의 Newey-West(HAC) 표준오차."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(s, 0) / n))


def ljung_box_p(x: np.ndarray, lags: int = 10) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n <= lags + 2:
        return np.nan
    e = x - x.mean()
    d = e @ e
    if d == 0:
        return np.nan
    q = 0.0
    for k in range(1, lags + 1):
        r = (e[k:] @ e[:-k]) / d
        q += r * r / (n - k)
    q *= n * (n + 2)
    return float(1 - stats.chi2.cdf(q, lags))
