"""6단계: 투수 vs 타자 맞대결 예측.

실제 맞대결 기록은 쌍당 평균 1타석도 안 되므로(희소성), '기록'이 아닌 '특성'으로 예측합니다.
- 기준선: 리그 평균, Log5(odds-ratio) 공식
- 유형 대 유형: 투수 유형 × 타자 유형 교차 빈도 (라플라스 평활)
- 다항 로지스틱 회귀
- LightGBM 다중분류 (최종 모델)
- 행렬 분해(ALS): LightGBM이 못 잡은 '궁합' 잔차를 투수·타자 잠재벡터로 학습

누수 방지: 시즌 s의 타석은 시즌 s-1의 특성으로 예측. 검증은 TEST_SEASON 전체.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from ..config import PROCESSED, TEST_SEASON
from .common import load, save, save_json

OUTCOMES = ["K", "BB", "1B", "XBH", "HR", "OUT"]
WOBA_W = np.array([0.0, 0.70, 0.88, 1.27, 2.03, 0.0])
SHRINK_PA = 150
GROUPS = ["패스트볼", "브레이킹", "오프스피드"]
P_EXTRA = ["Whiff%", "Chase%", "Zone%", "GB%", "HardHit%", "패스트볼 평균구속", "구위 지수"]
B_EXTRA = ["Chase%", "Whiff%", "Z-Contact%", "Barrel%", "HardHit%", "평균 발사각"]


# ------------------------------------------------------------------ 특성 테이블
def _shrunk_rates(pa: pd.DataFrame, key: str) -> pd.DataFrame:
    reg = pa[~pa["is_post"] & pa["outcome"].notna()]
    cnt = reg.groupby([key, "season", "outcome"]).size().unstack(fill_value=0).reindex(columns=OUTCOMES, fill_value=0)
    n = cnt.sum(axis=1)
    lg = cnt.groupby(level="season").sum()
    lg = lg.div(lg.sum(axis=1), axis=0)
    prior = lg.reindex(cnt.index.get_level_values("season")).values
    rates = (cnt.values + SHRINK_PA * prior) / (n.values[:, None] + SHRINK_PA)
    out = pd.DataFrame(rates, index=cnt.index, columns=[f"r_{o}" for o in OUTCOMES])
    out["n_pa"] = n.values
    return out.reset_index(), lg


def feature_tables():
    pa = load("pa")
    pr, lg = _shrunk_rates(pa, "pitcher")
    br, _ = _shrunk_rates(pa, "batter")
    ps, bs = load("pitcher_season"), load("batter_season")
    st = load("pitcher_stuff")
    if len(st):
        ps = ps.merge(st, on=["pitcher", "season"], how="left")
    pcols = [c for c in P_EXTRA if c in ps]
    pf = pr.merge(ps[["pitcher", "season"] + pcols], on=["pitcher", "season"], how="left")
    # 좌·우타자별 구종 그룹 구사율
    ar = load("pitcher_arsenal")
    from ..config import PITCH_GROUP
    ar["grp"] = ar["pitch_type"].map(PITCH_GROUP)
    for side, col in (("L", "좌타 구사율"), ("R", "우타 구사율")):
        u = ar.groupby(["pitcher", "season", "grp"])[col].sum().unstack().reindex(columns=GROUPS).fillna(0)
        u.columns = [f"use_{side}_{g}" for g in GROUPS]
        pf = pf.merge(u.reset_index(), on=["pitcher", "season"], how="left")
    bcols = [c for c in B_EXTRA if c in bs]
    bf = br.merge(bs[["batter", "season"] + bcols], on=["batter", "season"], how="left")
    vg = load("batter_vs_group")
    vg = vg[vg["key"].isin(GROUPS)]
    lgv = vg.groupby(["season", "key"]).apply(lambda d: np.average(d["xwOBA"].fillna(.32), weights=d["PA"].clip(1)),
                                                include_groups=False)
    vg = vg.assign(lgx=[lgv.get((s, k), .32) for s, k in zip(vg["season"], vg["key"])])
    vg["xw_shr"] = (vg["xwOBA"].fillna(vg["lgx"]) * vg["PA"] + 60 * vg["lgx"]) / (vg["PA"] + 60)
    v = vg.pivot_table(index=["batter", "season"], columns="key", values="xw_shr").reindex(columns=GROUPS)
    v.columns = [f"xw_vs_{g}" for g in GROUPS]
    bf = bf.merge(v.reset_index(), on=["batter", "season"], how="left")
    pc = load("pitcher_clusters")
    bc = load("batter_clusters")
    if len(pc):
        pf = pf.merge(pc[["pitcher", "season", "유형"]].rename(columns={"유형": "p_type"}), on=["pitcher", "season"], how="left")
    if len(bc):
        bf = bf.merge(bc[["batter", "season", "유형"]].rename(columns={"유형": "b_type"}), on=["batter", "season"], how="left")
    return pa, pf, bf, lg


def _prior_merge(rows: pd.DataFrame, pf: pd.DataFrame, bf: pd.DataFrame, lg: pd.DataFrame) -> pd.DataFrame:
    """각 타석에 '직전 시즌' 특성 붙이기. 직전 시즌 기록이 없으면 리그 평균 + 신인 플래그."""
    r = rows.copy()
    r["prior"] = r["season"] - 1
    p = pf.rename(columns={c: f"p_{c}" for c in pf.columns if c not in ("pitcher", "season")}).rename(columns={"season": "prior"})
    b = bf.rename(columns={c: f"b_{c}" for c in bf.columns if c not in ("batter", "season")}).rename(columns={"season": "prior"})
    r = r.merge(p, on=["pitcher", "prior"], how="left").merge(b, on=["batter", "prior"], how="left")
    r["p_new"] = r["p_n_pa"].isna().astype(int)
    r["b_new"] = r["b_n_pa"].isna().astype(int)
    for o in OUTCOMES:
        lgo = r["prior"].map(lg[o]) if o in lg else np.nan
        r[f"lg_{o}"] = lgo
        r[f"p_r_{o}"] = r[f"p_r_{o}"].fillna(lgo)
        r[f"b_r_{o}"] = r[f"b_r_{o}"].fillna(lgo)
    return r


def add_matchup_features(r: pd.DataFrame) -> pd.DataFrame:
    """Log5 확률, 플래툰, 구종 조합 기대 xwOBA. 앱 시뮬레이터에서도 사용."""
    r = r.copy()
    num = np.column_stack([r[f"p_r_{o}"] * r[f"b_r_{o}"] / r[f"lg_{o}"].clip(1e-6) for o in OUTCOMES])
    num = num / num.sum(axis=1, keepdims=True)
    for i, o in enumerate(OUTCOMES):
        r[f"log5_{o}"] = num[:, i]
    r["log5_wOBA"] = num @ WOBA_W
    r["platoon_adv"] = (r["stand"] != r["p_throws"]).astype(int)
    combo = np.zeros(len(r))
    for g in GROUPS:
        uL, uR = r.get(f"p_use_L_{g}"), r.get(f"p_use_R_{g}")
        if uL is None:
            continue
        use = np.where(r["stand"] == "L", uL, uR)
        combo += np.nan_to_num(use, nan=1 / 3) * r.get(f"b_xw_vs_{g}", pd.Series(.32, index=r.index)).fillna(.32).values
    r["구종조합_xwOBA"] = combo
    return r


def model_columns(r: pd.DataFrame) -> list[str]:
    drop = {"pitcher", "batter", "season", "prior", "game_date", "is_post", "outcome", "woba", "xwoba", "is_hit",
            "is_ab", "stand", "p_throws", "y", "p_p_type", "b_b_type"}
    return [c for c in r.columns if c not in drop and r[c].dtype != object and not c.startswith("lg_")]


# ------------------------------------------------------------------ 행렬 분해 (ALS)
def als(pairs: pd.DataFrame, dim: int = 4, lam: float = 8.0, iters: int = 15):
    pi = {p: i for i, p in enumerate(pairs["pitcher"].unique())}
    bi = {b: i for i, b in enumerate(pairs["batter"].unique())}
    R = np.zeros((len(pi), len(bi)))
    C = np.zeros_like(R)
    R[pairs["pitcher"].map(pi), pairs["batter"].map(bi)] = pairs["resid"].values
    C[pairs["pitcher"].map(pi), pairs["batter"].map(bi)] = pairs["n"].values
    rng = np.random.default_rng(0)
    U = rng.normal(0, .01, (len(pi), dim))
    V = rng.normal(0, .01, (len(bi), dim))
    I = np.eye(dim)
    for _ in range(iters):
        for i in range(len(pi)):
            c = C[i]
            U[i] = np.linalg.solve((V.T * c) @ V + lam * I, (V.T * c) @ R[i])
        for j in range(len(bi)):
            c = C[:, j]
            V[j] = np.linalg.solve((U.T * c) @ U + lam * I, (U.T * c) @ R[:, j])
    return pd.DataFrame(U, index=list(pi)), pd.DataFrame(V, index=list(bi))


def _mf_predict(U, V, p, b):
    out = np.zeros(len(p))
    ok = np.isin(p, U.index) & np.isin(b, V.index)
    if ok.any():
        out[ok] = np.einsum("ij,ij->i", U.loc[p[ok]].values, V.loc[b[ok]].values)
    return out


def _brier(P, y):
    Y = np.eye(len(OUTCOMES))[y]
    return float(((P - Y) ** 2).sum(axis=1).mean())


# ------------------------------------------------------------------ 실행
def main():
    print("[matchup] 특성 테이블")
    pa, pf, bf, lg = feature_tables()
    seasons = sorted(pa["season"].unique())
    test_s = TEST_SEASON if TEST_SEASON in seasons else seasons[-1]
    rows = pa[~pa["is_post"] & pa["outcome"].notna() & (pa["season"] > seasons[0])]
    r = add_matchup_features(_prior_merge(rows, pf, bf, lg))
    r["y"] = r["outcome"].map({o: i for i, o in enumerate(OUTCOMES)})
    tr, te = r[r["season"] < test_s], r[r["season"] == test_s]
    cols = model_columns(r)
    print(f"  학습 {len(tr):,}타석 ({tr['season'].min()}–{tr['season'].max()}), 검증 {len(te):,}타석 ({test_s}), 특성 {len(cols)}개")
    res = {}
    # 1) 리그 평균
    base = np.tile(np.bincount(tr["y"], minlength=6) / len(tr), (len(te), 1))
    res["리그 평균"] = base
    # 2) Log5
    res["Log5"] = te[[f"log5_{o}" for o in OUTCOMES]].values
    # 3) 유형 대 유형
    if "p_p_type" in r and r["p_p_type"].notna().any():
        tt = tr.dropna(subset=["p_p_type", "b_b_type"])
        cnt = tt.groupby(["p_p_type", "b_b_type", "y"]).size().unstack(fill_value=0).reindex(columns=range(6), fill_value=0)
        prior = np.bincount(tr["y"], minlength=6) / len(tr)
        sm = (cnt + 50 * prior)
        sm = sm.div(sm.sum(axis=1), axis=0)
        key = list(zip(te["p_p_type"], te["b_b_type"]))
        res["유형 대 유형"] = np.array([sm.loc[k].values if k in sm.index else prior for k in key])
        type_tab = tt.groupby(["p_p_type", "b_b_type"]).agg(
            PA=("y", "size"), K=("y", lambda s: (s == 0).mean()), BB=("y", lambda s: (s == 1).mean()),
            HR=("y", lambda s: (s == 4).mean()), xwOBA=("xwoba", "mean")).reset_index()
        type_tab.columns = ["투수 유형", "타자 유형", "PA", "K%", "BB%", "HR%", "xwOBA"]
        save("matchup_type_table", type_tab)
    # 4) 다항 로지스틱
    sc = StandardScaler().fit(tr[cols].fillna(0))
    lr = LogisticRegression(max_iter=300, C=0.5).fit(sc.transform(tr[cols].fillna(0)), tr["y"])
    res["다항 로지스틱"] = lr.predict_proba(sc.transform(te[cols].fillna(0)))
    # 5) LightGBM
    params = dict(objective="multiclass", num_class=6, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)
    # 조기 종료용 검증셋은 학습 기간의 마지막 20% (검증 시즌은 절대 보지 않음)
    cut = tr["game_date"].quantile(0.8)
    fit, val = tr[tr["game_date"] < cut], tr[tr["game_date"] >= cut]
    dtr = lgb.Dataset(fit[cols], fit["y"])
    dva = lgb.Dataset(val[cols], val["y"])
    booster = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    best_iter = booster.best_iteration or 200
    res["LightGBM"] = booster.predict(te[cols], num_iteration=best_iter)
    # 6) LightGBM + 행렬 분해 (xwOBA 잔차)
    tr_pred = booster.predict(tr[cols], num_iteration=best_iter) @ WOBA_W
    pairs = tr.assign(resid=tr["xwoba"].fillna(tr["woba"]) - tr_pred).groupby(["pitcher", "batter"]).agg(
        resid=("resid", "mean"), n=("resid", "size")).reset_index()
    U, V = als(pairs)
    te_xw = res["LightGBM"] @ WOBA_W
    te_mf = te_xw + _mf_predict(U, V, te["pitcher"].values, te["batter"].values)
    actual = te["xwoba"].fillna(te["woba"]).values
    # 평가
    comp = []
    ll0 = log_loss(te["y"], base, labels=range(6))
    for name, P in res.items():
        P = np.clip(P, 1e-6, 1)
        P = P / P.sum(axis=1, keepdims=True)
        ll = log_loss(te["y"], P, labels=range(6))
        comp.append({"모델": name, "로그 손실": ll, "Brier": _brier(P, te["y"].values),
                     "개선율(vs 리그 평균)": 1 - ll / ll0,
                     "xwOBA RMSE": float(np.sqrt(np.mean((P @ WOBA_W - actual) ** 2)))})
    comp.append({"모델": "LightGBM + 행렬분해", "로그 손실": np.nan, "Brier": np.nan, "개선율(vs 리그 평균)": np.nan,
                 "xwOBA RMSE": float(np.sqrt(np.mean((te_mf - actual) ** 2)))})
    comp = pd.DataFrame(comp)
    save("matchup_model_comparison", comp)
    print(comp.round(4).to_string(index=False))
    # 보정(calibration): 예측 기대 wOBA 10분위별 실제
    cal = pd.DataFrame({"pred": te_xw, "actual": actual})
    cal["bin"] = pd.qcut(cal["pred"], 10, labels=False, duplicates="drop")
    save("matchup_calibration", cal.groupby("bin").agg(예측=("pred", "mean"), 실제=("actual", "mean"), n=("pred", "size")).reset_index())
    imp = pd.DataFrame({"특성": cols, "중요도(gain)": booster.feature_importance("gain")}).sort_values("중요도(gain)", ascending=False)
    save("matchup_importance", imp)
    # 많이 만난 쌍(≥ 8타석)에서 예측 vs 실제
    tp = te.assign(pred=te_xw, act=actual).groupby(["pitcher", "batter"]).agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"))
    tp = tp[tp["n"] >= 8]
    # 앱용 최종 모델: 전체 시즌으로 재학습
    print("[matchup] 앱용 최종 모델 재학습 (전체 시즌)")
    final = lgb.train(params, lgb.Dataset(r[cols], r["y"]), best_iter)
    final.save_model(str(PROCESSED / "matchup_lgb.txt"))
    allp = r.assign(resid=r["xwoba"].fillna(r["woba"]) - final.predict(r[cols]) @ WOBA_W).groupby(["pitcher", "batter"]).agg(
        resid=("resid", "mean"), n=("resid", "size")).reset_index()
    U2, V2 = als(allp)
    save("matchup_mf_pitcher", U2.reset_index(names="pitcher").rename(columns=str))
    save("matchup_mf_batter", V2.reset_index(names="batter").rename(columns=str))
    save("matchup_features_pitcher", pf)
    save("matchup_features_batter", bf)
    save("matchup_league_rates", lg.reset_index())
    # 실제 맞대결 기록 (정규시즌 누적)
    reg = pa[~pa["is_post"] & pa["outcome"].notna()]
    h2h = reg.groupby(["pitcher", "batter"]).agg(
        PA=("outcome", "size"), AB=("is_ab", "sum"), H=("is_hit", "sum"), HR=("outcome", lambda s: (s == "HR").sum()),
        K=("outcome", lambda s: (s == "K").sum()), BB=("outcome", lambda s: (s == "BB").sum()),
        wOBA=("woba", "mean"), xwOBA=("xwoba", "mean")).reset_index()
    save("h2h", h2h)
    n_p, n_b = pa["pitcher"].nunique(), pa["batter"].nunique()
    meta = {"features": cols, "best_iter": int(best_iter), "test_season": int(test_s), "train_pa": len(tr), "test_pa": len(te),
            "possible_pairs_per_season": int(pa.groupby("season")["pitcher"].nunique().mean() * pa.groupby("season")["batter"].nunique().mean()),
            "pa_per_season": float(reg.groupby("season").size().mean()),
            "observed_pairs": int(len(h2h)), "pairs_ge_20pa": int((h2h["PA"] >= 20).sum()),
            "frequent_pairs_corr": float(tp["pred"].corr(tp["act"])) if len(tp) > 10 else None,
            "frequent_pairs_n": int(len(tp)), "n_pitchers": int(n_p), "n_batters": int(n_b)}
    save_json("matchup_meta", meta)


# ------------------------------------------------------------------ 앱 시뮬레이터용
def predict_pairs(req: pd.DataFrame, pf: pd.DataFrame, bf: pd.DataFrame, lg: pd.DataFrame, booster, cols: list[str]):
    """req: pitcher, batter, stand, p_throws, season(=특성 시즌 + 1). 반환: (LightGBM 확률, Log5 확률, 특성 행)."""
    lg = lg.set_index("season") if "season" in lg.columns else lg
    r = add_matchup_features(_prior_merge(req, pf, bf, lg))
    for c in cols:
        if c not in r:
            r[c] = np.nan
    P = booster.predict(r[cols])
    L = r[[f"log5_{o}" for o in OUTCOMES]].values
    return P, L, r
