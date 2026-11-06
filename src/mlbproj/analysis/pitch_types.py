"""투수 유형: '패스트볼 × 주력 보조구종' 조합형 + 5가지 분석.

유형 정의 (규칙): 가장 많이 던진 패스트볼(포심/싱커/커터) × 가장 많이 던진 비패스트볼 구종
  포심-슬라이더형 / 포심-스위퍼형 / 포심-체인지업형 / 포심-스플리터형 / 포심-커브형
  싱커-슬라이더형 / 싱커-체인지업형 / 싱커-커브형 / 커터형
태그: 강속구(패스트볼 97mph+), 하이 포심(포심 수직무브 17in+), 땅볼형(GB 50%+),
      피치 구성(구사율 12% 이상 구종 수: 2피치/3피치/다양형 4개+), 선발/불펜

분석
 1) 계층적 군집(Ward) 덴드로그램: 9개 유형을 성향으로 다시 묶어 상위 그룹 도출
 2) 선형판별분석(LDA): 구종 비율 없이 '공의 물리 특성'만으로 유형을 맞힐 수 있나 (GroupKFold)
 3) Kruskal-Wallis + Dunn 사후검정(FDR): 유형별 성적 차이가 진짜인가, 효과크기 ε²
 4) 대응분석(CA): 유형 × 타석 결과 교차표를 한 지도에
 5) 유형 전환: 시즌 간 유형 변화 흐름 + 이중차분형 비교(바꾼 투수 vs 안 바꾼 투수)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..config import PITCH_NAME_KO
from .common import bh_fdr, load, save, save_json

FB ={"FF": "포심", "SI": "싱커", "FC": "커터"}
SEC = {"SL": "슬라이더", "ST": "스위퍼", "SV": "스위퍼", "CH": "체인지업", "FS": "스플리터", "FO": "스플리터",
       "CU": "커브", "KC": "커브", "CS": "커브", "SC": "체인지업", "KN": "기타", "EP": "기타"}
TYPE_ORDER = ["포심-슬라이더형", "포심-스위퍼형", "포심-커브형", "포심-체인지업형", "포심-스플리터형",
              "싱커-슬라이더형", "싱커-커브형", "싱커-체인지업형", "커터형"]
PROFILE = ["K%", "BB%", "Whiff%", "Chase%", "GB%", "HardHit%", "xwOBA", "패스트볼 평균구속", "패스트볼 수직무브",
           "패스트볼 수평무브", "구종 수"]
TEST_METRICS = ["K%", "BB%", "Whiff%", "GB%", "xwOBA", "HardHit%", "RV/100"]


def type_name(fb: str | None, sec: str | None) -> str:
    if fb == "FC" or fb is None:
        return "커터형" if fb == "FC" else "기타"
    f, s = FB[fb], SEC.get(sec, "기타")
    if f == "포심":
        return {"슬라이더": "포심-슬라이더형", "스위퍼": "포심-스위퍼형", "체인지업": "포심-체인지업형",
                "스플리터": "포심-스플리터형", "커브": "포심-커브형"}.get(s, "포심-슬라이더형")
    return {"슬라이더": "싱커-슬라이더형", "스위퍼": "싱커-슬라이더형", "체인지업": "싱커-체인지업형",
            "스플리터": "싱커-체인지업형", "커브": "싱커-커브형"}.get(s, "싱커-슬라이더형")


def build_types() -> pd.DataFrame:
    ar, ps = load("pitcher_arsenal"), load("pitcher_season")
    q = ps[ps["qualified"]][["pitcher", "season", "name", "패스트볼 평균구속", "GB%", "선발", "경기"]]
    ar = ar.merge(q[["pitcher", "season"]], on=["pitcher", "season"])
    rows = []
    for (p, s), g in ar.groupby(["pitcher", "season"]):
        g = g.sort_values("구사율", ascending=False)
        fbs = g[g["pitch_type"].isin(FB)]
        secs = g[~g["pitch_type"].isin(FB)]
        fb = fbs.iloc[0] if len(fbs) else None
        sec = secs.iloc[0] if len(secs) else None
        ff = g[g["pitch_type"] == "FF"]
        use = g["구사율"].values
        top2 = use[:2].sum()
        core = g[g["구사율"] >= 0.12]
        n_core = len(core)
        mix = "2피치" if n_core <= 2 else ("3피치" if n_core == 3 else "다양형")   # 구사율 12% 이상 구종 수
        comp = "+".join(PITCH_NAME_KO.get(x, x) for x in (g.head(2)["pitch_type"] if mix == "2피치" else core["pitch_type"]))
        rows.append({"pitcher": p, "season": s, "패스트볼": fb["pitch_type"] if fb is not None else None,
                     "보조구종": sec["pitch_type"] if sec is not None else None,
                     "유형": type_name(fb["pitch_type"] if fb is not None else None, sec["pitch_type"] if sec is not None else None),
                     "피치 구성": mix, "구성": comp,
                     "패스트볼 수직무브": fb["수직무브"] if fb is not None else np.nan,
                     "패스트볼 수평무브": fb["수평무브"] if fb is not None else np.nan,
                     "포심 수직무브": ff["수직무브"].iloc[0] if len(ff) else np.nan,
                     "구종 수": n_core})
    t = pd.DataFrame(rows).merge(q, on=["pitcher", "season"], how="left")
    tags = []
    for _, r in t.iterrows():
        tg = []
        if r["패스트볼 평균구속"] >= 97:
            tg.append("강속구")
        if r["포심 수직무브"] >= 17:
            tg.append("하이 포심")
        if r["GB%"] >= 0.50:
            tg.append("땅볼형")
        tags.append(" · ".join(tg))
    t["태그"] = tags
    t["역할"] = np.where((t["선발"].fillna(0) / t["경기"].replace(0, np.nan)).fillna(0) >= 0.5, "선발", "불펜")
    t["표시"] = t["역할"] + " · " + t["유형"] + np.where(t["태그"] != "", " · " + t["태그"], "") + " · " + t["피치 구성"] + " (" + t["구성"] + ")"
    return t


def hierarchy(d: pd.DataFrame):
    cols = [c for c in PROFILE if c in d]
    Z = d[cols].astype(float)
    Z = (Z - Z.mean()) / Z.std()
    prof = Z.groupby(d["유형"]).mean().reindex([x for x in TYPE_ORDER if x in set(d["유형"])])
    L = linkage(prof.values, method="ward")
    dn = dendrogram(L, labels=list(prof.index), no_plot=True)
    # 높이 간격이 가장 큰 곳에서 자름 (2~4 그룹)
    h = L[:, 2]
    gaps = np.diff(h)
    cand = {k: gaps[len(h) - k] for k in (2, 3, 4) if len(h) - k >= 0 and len(h) - k < len(gaps)}
    k = max(cand, key=cand.get) if cand else 3
    grp = fcluster(L, k, criterion="maxclust")
    gmap = dict(zip(prof.index, grp))
    # 그룹 이름: 그룹 안 유형들의 공통 패스트볼
    names = {}
    for g in sorted(set(grp)):
        members = [t for t, x in gmap.items() if x == g]
        fbs = {m.split("-")[0].replace("형", "") for m in members}
        names[g] = (" / ".join(sorted(fbs)) + " 계열") if len(fbs) <= 2 else "혼합 계열"
    seen = {}
    for g in names:
        n = names[g]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[g] = f"{n} {seen[n]}"
    groups = pd.DataFrame({"유형": list(gmap), "상위 그룹": [names[gmap[t]] for t in gmap]})
    dd = pd.DataFrame({"icoord": [list(map(float, x)) for x in dn["icoord"]], "dcoord": [list(map(float, x)) for x in dn["dcoord"]]})
    return prof.reset_index(), groups, dd, dn["ivl"], k


def lda(d: pd.DataFrame, ar: pd.DataFrame):
    # 물리 특성만: 패스트볼 + 보조구종의 구속·회전·무브먼트 + 릴리스 (구사율 제외)
    a = ar.set_index(["pitcher", "season", "pitch_type"])
    rows = []
    for _, r in d.iterrows():
        try:
            f = a.loc[(r["pitcher"], r["season"], r["패스트볼"])]
            s_ = a.loc[(r["pitcher"], r["season"], r["보조구종"])]
        except KeyError:
            rows.append(None)
            continue
        rows.append([f["구속"], f["회전수"], f["수직무브"], f["수평무브"], f["릴리스높이"], f["릴리스측면"], f["익스텐션"],
                     s_["구속"] - f["구속"], s_["회전수"], s_["수직무브"], s_["수평무브"]])
    feats = ["패스트볼 구속", "패스트볼 회전수", "패스트볼 수직무브", "패스트볼 수평무브", "릴리스 높이", "릴리스 측면", "익스텐션",
             "보조구종 구속차", "보조구종 회전수", "보조구종 수직무브", "보조구종 수평무브"]
    ok = [x is not None for x in rows]
    X = pd.DataFrame([x for x in rows if x is not None], columns=feats).astype(float)
    y = d.loc[ok, "유형"].values
    g = d.loc[ok, "pitcher"].values
    X = X.fillna(X.mean())
    Xs = StandardScaler().fit_transform(X)
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(Xs, y, g):
        m = LinearDiscriminantAnalysis().fit(Xs[tr], y[tr])
        pred[te] = m.predict(Xs[te])
    labels = [t for t in TYPE_ORDER if t in set(y)]
    cm = confusion_matrix(y, pred, labels=labels)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    acc = float((pred == y).mean())
    base = float(pd.Series(y).value_counts(normalize=True).max())
    full = LinearDiscriminantAnalysis().fit(Xs, y)
    xy = full.transform(Xs)[:, :2]
    coords = d.loc[ok, ["pitcher", "season", "name", "유형"]].reset_index(drop=True)
    coords["ld1"], coords["ld2"] = xy[:, 0], xy[:, 1]
    coef = pd.DataFrame(full.scalings_[:, :2], index=feats, columns=["ld1", "ld2"]).reset_index(names="변수")
    recall = pd.DataFrame({"유형": labels, "정확도(재현율)": np.diag(cmn), "n": cm.sum(axis=1)})
    conf = pd.DataFrame(cmn, index=labels, columns=labels).reset_index(names="실제")
    return coords, conf, recall, coef, {"accuracy": acc, "baseline": base, "explained": full.explained_variance_ratio_[:2].tolist()}


def dunn(values: pd.Series, groups: pd.Series, labels: list[str]) -> pd.DataFrame:
    r = values.rank()
    n = len(r)
    ties = values.value_counts()
    tie_c = (ties ** 3 - ties).sum() / (12 * (n - 1))
    out = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ra, rb = r[groups == a], r[groups == b]
            if len(ra) < 3 or len(rb) < 3:
                continue
            se = np.sqrt((n * (n + 1) / 12 - tie_c) * (1 / len(ra) + 1 / len(rb)))
            z = (ra.mean() - rb.mean()) / se
            out.append((a, b, z, 2 * stats.norm.sf(abs(z))))
    t = pd.DataFrame(out, columns=["유형1", "유형2", "z", "p"])
    t["q_FDR"] = bh_fdr(t["p"].values)
    return t


def group_tests(d: pd.DataFrame):
    # 같은 투수가 여러 시즌 들어가면 독립성 위배 → 투수별 가장 최근 시즌 1개만 사용
    last = d.sort_values("season").groupby("pitcher").tail(1)
    labels = [t for t in TYPE_ORDER if (last["유형"] == t).sum() >= 10]
    last = last[last["유형"].isin(labels)]
    kw, posthoc, ci = [], [], []
    for m in TEST_METRICS:
        if m not in last:
            continue
        x = last[[m, "유형"]].dropna()
        H, p = stats.kruskal(*[x.loc[x["유형"] == t, m] for t in labels])
        k, n = len(labels), len(x)
        kw.append({"지표": m, "H": H, "p": p, "효과크기 ε²": (H - k + 1) / (n - k), "n": n})
        dt = dunn(x[m], x["유형"], labels)
        dt["지표"] = m
        posthoc.append(dt)
        for t in labels:
            v = x.loc[x["유형"] == t, m]
            se = v.std(ddof=1) / np.sqrt(len(v))
            ci.append({"지표": m, "유형": t, "평균": v.mean(), "하한": v.mean() - 1.96 * se, "상한": v.mean() + 1.96 * se, "n": len(v)})
    kw = pd.DataFrame(kw)
    kw["q_FDR"] = bh_fdr(kw["p"].values)
    return kw, pd.concat(posthoc, ignore_index=True), pd.DataFrame(ci)


def correspondence(d: pd.DataFrame):
    pa = load("pa")
    if pa.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    pa = pa[~pa["is_post"] & pa["outcome"].notna()].merge(d[["pitcher", "season", "유형"]], on=["pitcher", "season"])
    bt = pa["bb_type"] if "bb_type" in pa else pd.Series(None, index=pa.index)
    res = np.select([pa["outcome"].eq("K"), pa["outcome"].eq("BB"), pa["outcome"].eq("HR"), pa["outcome"].eq("XBH"),
                     pa["outcome"].eq("1B"), pa["outcome"].eq("OUT") & bt.eq("ground_ball")],
                    ["삼진", "볼넷·사구", "홈런", "장타(2·3루타)", "1루타", "땅볼 아웃"], default="뜬공·라인 아웃")
    N = pd.crosstab(pa["유형"], res).reindex([t for t in TYPE_ORDER if t in set(pa["유형"])])
    P = N.values / N.values.sum()
    r, c = P.sum(axis=1), P.sum(axis=0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    F = (U * sv) / np.sqrt(r)[:, None]          # 행(유형) 주좌표
    G = (Vt.T * sv) / np.sqrt(c)[:, None]       # 열(결과) 주좌표
    inertia = sv ** 2
    rows = pd.DataFrame({"이름": N.index, "구분": "유형", "dim1": F[:, 0], "dim2": F[:, 1]})
    cols = pd.DataFrame({"이름": N.columns, "구분": "결과", "dim1": G[:, 0], "dim2": G[:, 1]})
    chi2 = stats.chi2_contingency(N.values)
    share = N.div(N.sum(axis=1), axis=0).reset_index(names="유형")
    return pd.concat([rows, cols], ignore_index=True), share, {
        "inertia_share": (inertia[:2] / inertia.sum()).tolist(), "chi2": float(chi2.statistic), "chi2_p": float(chi2.pvalue),
        "cramers_v": float(np.sqrt(chi2.statistic / (N.values.sum() * (min(N.shape) - 1)))), "n_pa": int(N.values.sum())}


def transitions(d: pd.DataFrame):
    ps = load("pitcher_season")
    a = d[["pitcher", "season", "유형"]].merge(ps[["pitcher", "season", "K%", "xwOBA", "Whiff%", "age"]], on=["pitcher", "season"])
    b = a.assign(season=a["season"] - 1)
    j = a.merge(b, on=["pitcher", "season"], suffixes=("", "_다음"))
    flow = j.groupby(["유형", "유형_다음"]).size().rename("n").reset_index()
    j["전환"] = j["유형"] != j["유형_다음"]
    for m in ("K%", "xwOBA", "Whiff%"):
        j[f"Δ{m}"] = j[f"{m}_다음"] - j[m]
    # 이중차분형 비교: 바꾼 투수의 변화 − 안 바꾼 투수의 변화, 직전 성적(평균회귀)과 나이 통제
    did = []
    for m in ("K%", "xwOBA", "Whiff%"):
        X = sm.add_constant(pd.DataFrame({"전환": j["전환"].astype(float), "직전값": j[m], "나이": j["age"]}).astype(float))
        ok = X.notna().all(axis=1) & j[f"Δ{m}"].notna()
        f = sm.OLS(j.loc[ok, f"Δ{m}"], X[ok]).fit(cov_type="HC1")
        did.append({"지표": m, "전환 효과": f.params["전환"], "표준오차": f.bse["전환"], "p": f.pvalues["전환"],
                    "전환 투수 평균 변화": j.loc[j["전환"], f"Δ{m}"].mean(), "유지 투수 평균 변화": j.loc[~j["전환"], f"Δ{m}"].mean(),
                    "전환 n": int(j["전환"].sum()), "유지 n": int((~j["전환"]).sum())})
    # 대표 사례: 포심-슬라이더 → 포심-스위퍼 (스위퍼 장착)
    sw = j[j["유형"] == "포심-슬라이더형"]
    sw = sw.assign(스위퍼전환=sw["유형_다음"] == "포심-스위퍼형")
    sweeper = {}
    if sw["스위퍼전환"].sum() >= 10:
        for m in ("K%", "xwOBA"):
            X = sm.add_constant(pd.DataFrame({"스위퍼전환": sw["스위퍼전환"].astype(float), "직전값": sw[m]}).astype(float))
            ok = X.notna().all(axis=1) & sw[f"Δ{m}"].notna()
            f = sm.OLS(sw.loc[ok, f"Δ{m}"], X[ok]).fit(cov_type="HC1")
            sweeper[m] = {"효과": float(f.params["스위퍼전환"]), "p": float(f.pvalues["스위퍼전환"]), "n": int(sw["스위퍼전환"].sum())}
    trend = d.groupby(["season", "유형"]).size().rename("n").reset_index()
    trend["비율"] = trend["n"] / trend.groupby("season")["n"].transform("sum")
    return flow, pd.DataFrame(did), trend, {"switch_rate": float(j["전환"].mean()), "sweeper": sweeper}


def main():
    print("[pitch_types] 규칙 유형")
    d = build_types()
    d = d[d["유형"] != "기타"]
    ps = load("pitcher_season")
    full = d.merge(ps[["pitcher", "season"] + [c for c in TEST_METRICS + ["age"] if c in ps and c not in d]], on=["pitcher", "season"], how="left")
    print("[pitch_types] 계층적 군집")
    prof, groups, dd, leaves, k = hierarchy(full)
    d = d.merge(groups, on="유형", how="left")
    save("pitcher_types", d)
    save("ptype_profile", prof)
    save("ptype_groups", groups)
    save("ptype_dendrogram", dd.assign(ivl=[leaves] * len(dd)) if len(dd) else dd)
    print("[pitch_types] LDA")
    coords, conf, recall, coef, lm = lda(d, load("pitcher_arsenal"))
    save("ptype_lda", coords.merge(groups, on="유형", how="left"))
    save("ptype_confusion", conf)
    save("ptype_lda_recall", recall)
    save("ptype_lda_coef", coef)
    print("[pitch_types] Kruskal-Wallis")
    kw, dn, ci = group_tests(full)
    save("ptype_kw", kw)
    save("ptype_dunn", dn)
    save("ptype_ci", ci)
    print("[pitch_types] 대응분석")
    ca, share, cam = correspondence(d)
    if len(ca):
        save("ptype_ca", ca)
        save("ptype_outcome_share", share)
    print("[pitch_types] 유형 전환")
    flow, did, trend, tm = transitions(full)
    save("ptype_flow", flow)
    save("ptype_did", did)
    save("ptype_trend", trend)
    cl = load("pitcher_clusters")
    ari = None
    if len(cl):
        m = d.merge(cl[["pitcher", "season", "유형"]].rename(columns={"유형": "kmeans"}), on=["pitcher", "season"])
        ari = float(adjusted_rand_score(m["유형"], m["kmeans"]))
        save("ptype_vs_kmeans", pd.crosstab(m["유형"], m["kmeans"]).reset_index())
    meta = {"n": len(d), "k_groups": int(k), "lda": lm, "ca": cam, "transition": tm, "ari_vs_kmeans": ari,
            "counts_latest": d[d["season"] == d["season"].max()]["유형"].value_counts().to_dict()}
    save_json("ptype_meta", meta)
    print(f"  유형 {d['유형'].nunique()}개, 상위 그룹 {k}개, LDA 정확도 {lm['accuracy']:.2f} (기준 {lm['baseline']:.2f}), "
          f"CA 설명력 {sum(cam.get('inertia_share', [0])):.0%}, 전환율 {tm['switch_rate']:.0%}")
