"""9단계: 모형 검증 종합표 (포스터의 '위협 → 검증 → 결과' 표)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ..config import PROCESSED
from . import clustering
from .common import load, save


def _meta(name):
    f = PROCESSED / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def cluster_threshold_sensitivity() -> list[dict]:
    """최소 표본 기준을 올려도 유형 배정이 유지되는가 (ARI)."""
    out = []
    meta = _meta("clusters_meta")
    for kind, col, levels in (("pitcher", "pitches", (300, 600, 1000)), ("batter", "PA", (150, 300, 450))):
        if kind not in meta:
            continue
        X = clustering.pitcher_features() if kind == "pitcher" else clustering.batter_features()
        X = X[meta[kind]["features"]]
        base = load(f"{kind}_clusters").set_index([kind, "season"])["유형"]
        size = load(f"{kind}_season").set_index([kind, "season"])[col]
        for lv in levels[1:]:
            sub = X[size.reindex(X.index) >= lv]
            if len(sub) < 30:
                continue
            km = KMeans(meta[kind]["k"], n_init=10, random_state=0).fit(StandardScaler().fit_transform(sub))
            out.append({"kind": kind, "기준": lv, "n": len(sub), "ARI": adjusted_rand_score(base.reindex(sub.index), km.labels_)})
    return out


def main():
    cm, sm_, lm, mm, am, ag = (_meta(n) for n in ("clusters_meta", "shap_meta", "luck_meta", "matchup_meta", "abs_meta", "aging_meta"))
    sens = cluster_threshold_sensitivity()
    comp = load("matchup_model_comparison")
    its = load("abs_its")
    did = load("abs_did_summary")
    rows = []

    def add(threat, method, result):
        rows.append({"잠재 위협": threat, "검증 방법": method, "결과": result})

    for kind, lab in (("pitcher", "투수"), ("batter", "타자")):
        if kind in cm:
            rm = ", ".join(f"{n}({'완전공선' if v > 1e6 else f'{v:.1f}'})" for n, v in cm[kind]["vif_removed"]) or "없음"
            add(f"다중공선성 ({lab} 군집)", "VIF > 10 변수 단계적 제거", f"제거: {rm} → {len(cm[kind]['features'])}개 변수")
            add(f"군집 수 자의성 ({lab})", "엘보우·실루엣·Davies-Bouldin·GMM BIC 비교", f"k = {cm[kind]['k']}")
            add(f"군집 불안정성 ({lab})", "부트스트랩 10회 재군집 ARI", f"평균 {cm[kind]['bootstrap_ari_mean']:.2f} (최소 {cm[kind]['bootstrap_ari_min']:.2f})")
    for s in sens:
        add(f"표본 기준 자의성 ({'투수' if s['kind'] == 'pitcher' else '타자'})", f"최소 기준 {s['기준']}로 상향 후 재군집",
            f"ARI {s['ARI']:.2f} (n={s['n']})")
    if sm_:
        add("구위 모델 과적합", "투수 단위 GroupKFold 5겹 교차검증", f"CV R² = {sm_['stuff']['cv_r2']:.3f}")
        if sm_["stuff"].get("stuff_to_next_whiff_r") is not None:
            add("구위 지수의 예측력", "올해 구위 지수 ↔ 내년 헛스윙률 상관",
                f"r = {sm_['stuff']['stuff_to_next_whiff_r']:.2f} (헛스윙률 자기상관 {sm_['stuff']['whiff_to_next_whiff_r']:.2f})")
        add("타자 모델 과적합", "타자 단위 GroupKFold 5겹 교차검증", f"CV R² = {sm_['batter']['cv_r2']:.3f}")
    if lm:
        if "huber_vs_ols_spearman" in lm:
            add("이상치 (ERA 회귀)", "Huber 로버스트 회귀 vs OLS 잔차 순위 비교", f"Spearman ρ = {lm['huber_vs_ols_spearman']:.3f}")
        add("자기상관 (경기 순서)", "Ljung-Box(시차 10) → Newey-West HAC 표준오차",
            f"자기상관 유의: 투수 {lm['ljungbox_autocorr_share_p']:.0%}, 타자 {lm['ljungbox_autocorr_share_b']:.0%}")
        add("다중검정 (수백 명 동시 검정)", "Benjamini-Hochberg FDR 5%",
            f"투수 {lm['pitcher_naive_sig']}→{lm['pitcher_sig']}명, 타자 {lm['batter_naive_sig']}→{lm['batter_sig']}명 (보정 전→후)")
        if lm.get("luck_persistence_b") is not None:
            add("기대 성적이 일시적인가", "wOBA−xwOBA의 연도 간 상관 vs K% 연도 간 상관",
                f"갭 r = {lm['luck_persistence_b']:.2f} / K% r = {lm.get('skill_persistence_b_K') or float('nan'):.2f}")
    if mm:
        add("맞대결 표본 희소성", "실제 기록 대신 특성 기반 모델",
            f"가능한 쌍 ≈ {mm['possible_pairs_per_season']:,}/시즌 vs 타석 ≈ {mm['pa_per_season']:,.0f}/시즌")
        add("데이터 누수", "시즌 s 타석을 s−1 특성으로 예측, 조기 종료도 학습기간 안에서만", f"검증 시즌 {mm['test_season']} 완전 분리")
        if len(comp):
            lgbm = comp.set_index("모델")
            add("모델이 단순 공식보다 나은가", "Log5·리그평균 대비 로그 손실",
                f"LightGBM {lgbm.loc['LightGBM', '로그 손실']:.4f} vs Log5 {lgbm.loc['Log5', '로그 손실']:.4f} vs 평균 {lgbm.loc['리그 평균', '로그 손실']:.4f}")
    if len(its):
        r = its.set_index("지표")
        k = "경계 콜 스트라이크율"
        if k in r.index:
            add("ABS 효과가 추세·계절성 착시일 가능성", "단절 시계열 + 월 고정효과 + HAC",
                f"{k} {r.loc[k, 'ABS 효과(%p)']:+.2f}%p (p={r.loc[k, 'p']:.3f})")
    if len(did) > 1:
        add("ABS 이질성의 사전 추세", "도입 전 구간에 같은 분석 적용(위약 검정)",
            f"도입 p={did.iloc[0]['ΔBB% p']:.3f} / 위약 p={did.iloc[1]['ΔBB% p']:.3f}")
    if ag.get("pitcher_velo"):
        add("에이징 커브의 생존 편향", "혼합효과모형(선수 무작위효과) + 델타 방법 비교",
            f"투수 구속 정점 {ag['pitcher_velo'].get('peak_age') or float('nan'):.1f}세")
    t = pd.DataFrame(rows)
    save("validation_table", t)
    if sens:
        save("validation_sensitivity", pd.DataFrame(sens))
    print(t.to_string(index=False))
