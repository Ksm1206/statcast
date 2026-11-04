"""4단계: 투수·타자 유형 분류 (VIF → 표준화 → k 선택 → K-means / GMM → PCA) + 유사 선수."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .common import load, save, save_json, vif_filter

LABELS = "ABCDEFGHIJ"


def pitcher_features() -> pd.DataFrame:
    ps = load("pitcher_season")
    ar = load("pitcher_arsenal")
    ps = ps[ps["qualified"]]
    use = ar.pivot_table(index=["pitcher", "season"], columns="pitch_type", values="구사율", aggfunc="sum").fillna(0)
    grp = pd.DataFrame(index=use.index)
    g = lambda *c: use[[x for x in c if x in use]].sum(axis=1)  # noqa: E731
    grp["포심%"], grp["싱커%"], grp["커터%"] = g("FF"), g("SI"), g("FC")
    grp["슬라이더/스위퍼%"], grp["커브%"], grp["체인지업/스플리터%"] = g("SL", "ST", "SV"), g("CU", "KC"), g("CH", "FS")
    fb = ar[ar["pitch_type"].isin(["FF", "SI", "FC"])].sort_values("pitches").groupby(["pitcher", "season"]).tail(1)
    fb = fb.set_index(["pitcher", "season"])[["구속", "수직무브", "수평무브", "익스텐션", "릴리스높이"]]
    fb.columns = ["패스트볼 구속", "패스트볼 수직무브", "패스트볼 수평무브", "익스텐션", "릴리스 높이"]
    X = ps.set_index(["pitcher", "season"])[["Zone%", "Chase%", "Whiff%", "GB%", "K%", "BB%"]]
    X = X.join(grp, how="left").join(fb, how="left")
    X["구종 수"] = ps.set_index(["pitcher", "season"])["구종 수"]
    return X.fillna(X.mean())


def batter_features() -> pd.DataFrame:
    bs = load("batter_season")
    bs = bs[bs["qualified"]]
    cols = ["K%", "BB%", "Chase%", "Z-Contact%", "Whiff%", "평균 타구속도", "EV90", "Barrel%", "평균 발사각",
            "당겨치기%", "GB%", "SweetSpot%", "배트 스피드", "스윙 길이"]
    X = bs.set_index(["batter", "season"])[[c for c in cols if c in bs]]
    X = X.loc[:, X.notna().mean() > 0.6]   # 배트 스피드는 2024년부터라 결측 많으면 제외
    return X.fillna(X.mean())


def _choose_k(Z: np.ndarray, ks=range(3, 9)) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        gm = GaussianMixture(k, random_state=0, covariance_type="diag").fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_, sample_size=min(4000, len(Z)), random_state=0),
                     "davies_bouldin": davies_bouldin_score(Z, km.labels_), "gmm_bic": gm.bic(Z)})
    t = pd.DataFrame(rows)
    # 실루엣 최대, 차이가 0.01 이내면 작은 k (해석 용이)
    best = t["silhouette"].max()
    k = int(t[t["silhouette"] >= best - 0.01]["k"].min())
    return k, t


def _describe(profile: pd.DataFrame) -> dict[str, str]:
    desc = {}
    for c, row in profile.iterrows():
        hi = row.sort_values(ascending=False)
        parts = [f"높은 {i}" for i in hi.index[:2] if hi[i] > 0.4]
        parts += [f"낮은 {i}" for i in hi.index[::-1][:1] if hi[i] < -0.4]
        desc[c] = " · ".join(parts) if parts else "평균형"
    return desc


def run_one(kind: str):
    X = pitcher_features() if kind == "pitcher" else batter_features()
    keep, removed = vif_filter(X)
    scaler = StandardScaler().fit(X[keep])
    Z = scaler.transform(X[keep])
    k, ktable = _choose_k(Z)
    km = KMeans(k, n_init=20, random_state=0).fit(Z)
    lab = np.array([LABELS[i] for i in km.labels_])
    pca = PCA(2, random_state=0).fit(Z)
    xy = pca.transform(Z)
    out = X.reset_index()[[kind, "season"]].copy()
    out["유형"], out["pc1"], out["pc2"] = lab, xy[:, 0], xy[:, 1]
    gmm = GaussianMixture(k, random_state=0, covariance_type="diag").fit(Z)
    out["gmm_확신도"] = gmm.predict_proba(Z).max(axis=1)
    zdf = pd.DataFrame(Z, columns=keep)
    zdf["유형"] = lab
    profile = zdf.groupby("유형").mean()
    raw = X[keep].reset_index(drop=True).assign(유형=lab).groupby("유형").mean()
    desc = _describe(profile)
    sizes = pd.Series(lab).value_counts().to_dict()
    # 안정성: 부트스트랩 재표본 군집과의 ARI
    rng = np.random.default_rng(0)
    aris = []
    for _ in range(10):
        idx = rng.choice(len(Z), len(Z), replace=True)
        km2 = KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(Z[idx])
        aris.append(adjusted_rand_score(km.labels_, km2.predict(Z)))
    save(f"{kind}_clusters", out)
    save(f"{kind}_cluster_profile", profile.reset_index())
    save(f"{kind}_cluster_raw", raw.reset_index())
    save(f"{kind}_k_selection", ktable)
    loadings = pd.DataFrame(pca.components_.T, index=keep, columns=["pc1", "pc2"]).reset_index(names="변수")
    save(f"{kind}_pca_loadings", loadings)
    meta = {"k": k, "features": keep, "vif_removed": removed, "desc": desc, "sizes": sizes,
            "pca_var": pca.explained_variance_ratio_.tolist(), "bootstrap_ari_mean": float(np.mean(aris)),
            "bootstrap_ari_min": float(np.min(aris))}
    # 유사 선수 (같은 시즌, 코사인 유사도 Top 5)
    sims = []
    meta_idx = X.reset_index()[[kind, "season"]]
    for s, idx in meta_idx.groupby("season").groups.items():
        idx = np.array(list(idx))
        V = Z[idx]
        V = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-9)
        S = V @ V.T
        np.fill_diagonal(S, -1)
        top = np.argsort(-S, axis=1)[:, :5]
        for i, row in enumerate(top):
            for rank, j in enumerate(row):
                sims.append((int(meta_idx.iloc[idx[i]][kind]), s, rank + 1, int(meta_idx.iloc[idx[j]][kind]), float(S[i, j])))
    save(f"{kind}_similar", pd.DataFrame(sims, columns=[kind, "season", "순위", "유사선수", "유사도"]))
    return meta


def main():
    meta = {}
    for kind in ("pitcher", "batter"):
        print(f"[clustering] {kind}")
        meta[kind] = run_one(kind)
        print(f"  k={meta[kind]['k']}  VIF 제거={[r[0] for r in meta[kind]['vif_removed']]}  ARI={meta[kind]['bootstrap_ari_mean']:.2f}")
    save_json("clusters_meta", meta)
    return meta
