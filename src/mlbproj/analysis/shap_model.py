"""4단계(요인 분석): LightGBM + SHAP.

- 구위 모델: 구종의 물리 특성만으로 헛스윙률을 예측 → '구위 지수'(100 = 리그 평균)
- 타자 모델: 스윙·선구 특성으로 wOBA를 예측 → 어떤 능력이 생산성을 만드는가
SHAP 값은 LightGBM 내장 pred_contrib(TreeSHAP)로 계산합니다.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..config import MIN_PITCHES_PER_TYPE, PITCH_GROUP
from .common import load, save, save_json

STUFF_FEATS = ["구속", "회전수", "수직무브", "수평무브", "익스텐션", "릴리스높이", "릴리스측면",
               "패스트볼 대비 구속차", "수직무브 차", "수평무브 차", "is_fb", "is_breaking", "is_offspeed"]
# K%·BB%는 wOBA 계산에 직접 들어가는 성분이라 넣으면 '정답으로 정답을 설명'하게 됨 → 과정 지표만 사용
BAT_FEATS = ["Chase%", "Z-Contact%", "Whiff%", "Z-Swing%", "평균 발사각", "당겨치기%", "SweetSpot%", "배트 스피드",
             "스윙 길이", "EV90"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, seed=0)


def _cv_fit(X, y, w, groups, rounds=400):
    preds = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y.iloc[tr], weight=w.iloc[tr]), rounds)
        preds[te] = m.predict(X.iloc[te])
    r2 = 1 - np.average((y - preds) ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    model = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w), rounds)
    contrib = model.predict(X, pred_contrib=True)
    return model, preds, float(r2), contrib


def stuff_model():
    ar = load("pitcher_arsenal")
    ar = ar[(ar["pitches"] >= MIN_PITCHES_PER_TYPE) & ar["Whiff%"].notna()].copy()
    g = ar["pitch_type"].map(PITCH_GROUP)
    ar["is_fb"], ar["is_breaking"], ar["is_offspeed"] = (g == "패스트볼") * 1, (g == "브레이킹") * 1, (g == "오프스피드") * 1
    feats = [f for f in STUFF_FEATS if f in ar and ar[f].notna().mean() > .5]
    X = ar[feats].astype(float)
    y = ar["Whiff%"].astype(float)
    w = np.sqrt(ar["pitches"].astype(float))
    model, oof, r2, contrib = _cv_fit(X, y, w, ar["pitcher"])
    pred = model.predict(X)
    # 구종 그룹별 평균 대비 지수 (100 = 같은 구종 그룹의 리그 평균)
    ar["예측 헛스윙률"] = pred
    base = ar.groupby(g)["예측 헛스윙률"].transform(lambda s: np.average(s, weights=ar.loc[s.index, "pitches"]))
    ar["구위 지수"] = 100 * ar["예측 헛스윙률"] / base
    sh = pd.DataFrame(contrib[:, :-1], columns=feats)
    imp = sh.abs().mean().sort_values(ascending=False)
    out = ar[["pitcher", "season", "pitch_type", "pitches", "Whiff%", "예측 헛스윙률", "구위 지수"]].reset_index(drop=True)
    shap_rows = pd.concat([out[["pitcher", "season", "pitch_type"]], sh.add_prefix("shap_"), X.reset_index(drop=True).add_prefix("val_")], axis=1)
    save("stuff_scores", out)
    save("stuff_shap", shap_rows)
    # 투수별 구위 지수 (투구 수 가중 평균)
    p = out.groupby(["pitcher", "season"]).apply(lambda d: np.average(d["구위 지수"], weights=d["pitches"]), include_groups=False)
    save("pitcher_stuff", p.rename("구위 지수").reset_index())
    # 다음 시즌 안정성: 올해 구위 지수 ↔ 내년 실제 헛스윙률
    nxt = out.merge(out.assign(season=out["season"] - 1)[["pitcher", "season", "pitch_type", "Whiff%"]],
                    on=["pitcher", "season", "pitch_type"], suffixes=("", "_next"))
    stab = {"stuff_to_next_whiff_r": float(nxt["구위 지수"].corr(nxt["Whiff%_next"])) if len(nxt) > 10 else None,
            "whiff_to_next_whiff_r": float(nxt["Whiff%"].corr(nxt["Whiff%_next"])) if len(nxt) > 10 else None}
    return {"cv_r2": r2, "n": len(out), "importance": imp.to_dict(), **stab}


def batter_model():
    bs = load("batter_season")
    bs = bs[bs["qualified"] & bs["wOBA"].notna()].copy()
    feats = [f for f in BAT_FEATS if f in bs and bs[f].notna().mean() > .6]
    X = bs[feats].astype(float)
    y = bs["wOBA"].astype(float)
    w = np.sqrt(bs["PA"].astype(float))
    model, oof, r2, contrib = _cv_fit(X, y, w, bs["batter"], rounds=300)
    sh = pd.DataFrame(contrib[:, :-1], columns=feats)
    imp = sh.abs().mean().sort_values(ascending=False)
    rows = pd.concat([bs[["batter", "season"]].reset_index(drop=True), sh.add_prefix("shap_"),
                      X.reset_index(drop=True).add_prefix("val_")], axis=1)
    save("batter_shap", rows)
    return {"cv_r2": r2, "n": len(bs), "importance": imp.to_dict()}


def main():
    print("[shap] 구위 모델")
    a = stuff_model()
    print(f"  CV R²={a['cv_r2']:.3f}")
    print("[shap] 타자 모델")
    b = batter_model()
    print(f"  CV R²={b['cv_r2']:.3f}")
    save_json("shap_meta", {"stuff": a, "batter": b})
