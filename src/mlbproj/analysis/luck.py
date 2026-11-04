"""5단계: 기대 성적 (결과 vs 과정).

1) wOBA - xwOBA (타석 단위 차이)의 평균 → 정규근사 z, Newey-West(HAC) z, Ljung-Box, BH-FDR
2) Huber 로버스트 회귀: ERA ~ K% + BB% + GB% + HardHit% → 잔차 = 수비·운·시퀀싱 요인
   OLS 결과와 순위 상관으로 강건성 확인
3) 연도 간 지속성: 갭은 다음 해로 이어지지 않아야 함 (실력 지표 K%와 비교)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression, TheilSenRegressor
from sklearn.preprocessing import StandardScaler

from .common import bh_fdr, ljung_box_p, load, newey_west_se, save, save_json


def _player_luck(pa: pd.DataFrame, key: str, min_pa: int) -> pd.DataFrame:
    pa = pa[~pa["is_post"] & pa["xwoba"].notna() & pa["woba"].notna()].sort_values("game_date")
    pa = pa.assign(diff=pa["woba"] - pa["xwoba"])
    rows = []
    for (pid, s), g in pa.groupby([key, "season"]):
        n = len(g)
        if n < min_pa:
            continue
        d = g["diff"].values
        m, sd = d.mean(), d.std(ddof=1)
        se = sd / np.sqrt(n)
        se_hac = newey_west_se(d)
        rows.append((pid, s, n, g["woba"].mean(), g["xwoba"].mean(), m, m / se if se > 0 else np.nan,
                     m / se_hac if se_hac and se_hac > 0 else np.nan, ljung_box_p(d)))
    t = pd.DataFrame(rows, columns=[key, "season", "PA", "wOBA", "xwOBA", "차이", "z", "z_HAC", "LjungBox_p"])
    t["p"] = 2 * stats.norm.sf(t["z"].abs())
    t["p_HAC"] = 2 * stats.norm.sf(t["z_HAC"].abs())
    t["q_FDR"] = bh_fdr(t["p_HAC"].values)
    t["유의(FDR 5%)"] = t["q_FDR"] < 0.05
    # 95% 부트스트랩 근사 구간 (정규근사)
    t["CI_low"] = t["차이"] - 1.96 * t["차이"] / t["z_HAC"].replace(0, np.nan)
    t["CI_high"] = t["차이"] + 1.96 * t["차이"] / t["z_HAC"].replace(0, np.nan)
    return t


def _persistence(t: pd.DataFrame, key: str, col: str) -> float | None:
    nx = t.merge(t.assign(season=t["season"] - 1)[[key, "season", col]], on=[key, "season"], suffixes=("", "_next"))
    return float(nx[col].corr(nx[f"{col}_next"])) if len(nx) > 10 else None


def era_residuals() -> tuple[pd.DataFrame, dict]:
    ps = load("pitcher_season")
    feats = ["K%", "BB%", "GB%", "HardHit%"]
    d = ps[ps["qualified"] & ps["ERA"].notna() & (ps["IP"] >= 20)].dropna(subset=feats).copy()
    if len(d) < 20:
        return pd.DataFrame(), {}
    Z = StandardScaler().fit_transform(d[feats])
    w = d["IP"].values
    hub = HuberRegressor(epsilon=1.35, max_iter=500).fit(Z, d["ERA"], sample_weight=w)
    ols = LinearRegression().fit(Z, d["ERA"], sample_weight=w)
    d["기대 ERA"] = hub.predict(Z)
    d["ERA 잔차"] = d["ERA"] - d["기대 ERA"]
    d["ERA 잔차(OLS)"] = d["ERA"] - ols.predict(Z)
    ts = TheilSenRegressor(random_state=0, max_subpopulation=2000, n_jobs=-1).fit(Z, d["ERA"])
    d["ERA 잔차(Theil-Sen)"] = d["ERA"] - ts.predict(Z)
    rho = stats.spearmanr(d["ERA 잔차"], d["ERA 잔차(OLS)"]).statistic
    rho_ts = stats.spearmanr(d["ERA 잔차"], d["ERA 잔차(Theil-Sen)"]).statistic
    rho_ols_ts = stats.spearmanr(d["ERA 잔차(OLS)"], d["ERA 잔차(Theil-Sen)"]).statistic

    def top_overlap(a, b, k=20):  # 시즌별 '불운 상위 k명'이 얼마나 겹치나
        out = []
        for _, g in d.groupby("season"):
            A, B = set(g.nlargest(k, a)["pitcher"]), set(g.nlargest(k, b)["pitcher"])
            out.append(len(A & B) / k)
        return float(np.mean(out))
    # 회귀 진단: 잔차 vs 예측 + LOWESS, Q-Q
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(d["ERA 잔차"], d["기대 ERA"], frac=0.3, return_sorted=True)
    r = d["ERA 잔차"].values
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    q = pd.DataFrame({"이론 분위수": osm, "표본 분위수": (np.sort(r) - r.mean()) / r.std()})
    save("diag_lowess", pd.DataFrame({"기대 ERA": lw[:, 0], "LOWESS": lw[:, 1]}))
    save("diag_qq", q.iloc[:: max(1, len(q) // 600)])
    save("diag_resid", d[["기대 ERA", "ERA 잔차"]].sample(min(len(d), 2500), random_state=0))
    coef = pd.DataFrame({"변수": feats, "OLS": ols.coef_, "Huber": hub.coef_, "Theil-Sen": ts.coef_})
    save("diag_coef", coef)
    meta = {"theilsen_coef": dict(zip(feats, ts.coef_.round(3).tolist())),
            "huber_vs_theilsen_spearman": float(rho_ts), "ols_vs_theilsen_spearman": float(rho_ols_ts),
            "top20_overlap_huber_ols": top_overlap("ERA 잔차", "ERA 잔차(OLS)"),
            "top20_overlap_huber_ts": top_overlap("ERA 잔차", "ERA 잔차(Theil-Sen)"),
            "resid_skew": float(stats.skew(r)), "resid_excess_kurtosis": float(stats.kurtosis(r)),
            "lowess_max_abs": float(np.abs(lw[:, 1]).max()),
            "shapiro_p": float(stats.shapiro(r[:5000]).pvalue),
            "huber_coef": dict(zip(feats, hub.coef_.round(3).tolist())), "ols_coef": dict(zip(feats, ols.coef_.round(3).tolist())),
            "huber_vs_ols_spearman": float(rho), "n": len(d),
            "era_resid_persistence": _persistence(d.rename(columns={"pitcher": "pid"}), "pid", "ERA 잔차"),
            "k_persistence": _persistence(d.rename(columns={"pitcher": "pid"}), "pid", "K%")}
    return d[["pitcher", "season", "ERA", "기대 ERA", "ERA 잔차", "ERA 잔차(OLS)", "ERA 잔차(Theil-Sen)", "FIP", "IP"]], meta


def main():
    pa = load("pa")
    print("[luck] 투수")
    lp = _player_luck(pa, "pitcher", 150)
    print("[luck] 타자")
    lb = _player_luck(pa, "batter", 150)
    save("luck_pitcher", lp)
    save("luck_batter", lb)
    er, meta = era_residuals()
    if len(er):
        save("era_residuals", er)
    bs = load("batter_season")
    meta.update({
        "pitcher_sig": int(lp["유의(FDR 5%)"].sum()), "pitcher_n": len(lp),
        "batter_sig": int(lb["유의(FDR 5%)"].sum()), "batter_n": len(lb),
        "pitcher_naive_sig": int((lp["p"] < .05).sum()), "batter_naive_sig": int((lb["p"] < .05).sum()),
        "ljungbox_autocorr_share_p": float((lp["LjungBox_p"] < .05).mean()),
        "ljungbox_autocorr_share_b": float((lb["LjungBox_p"] < .05).mean()),
        "luck_persistence_b": _persistence(lb, "batter", "차이"),
        "luck_persistence_p": _persistence(lp, "pitcher", "차이"),
        "skill_persistence_b_K": _persistence(bs[bs["qualified"]], "batter", "K%"),
        "skill_persistence_b_xwoba": _persistence(bs[bs["qualified"]], "batter", "xwOBA"),
    })
    save_json("luck_meta", meta)
    print(f"  유의 투수 {meta['pitcher_sig']}/{meta['pitcher_n']}, 타자 {meta['batter_sig']}/{meta['batter_n']}")
