import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import radar, PCFG, CAT, DIV, INK2, MUTED, SURFACE, load, meta, names, note, require_data, setup

setup("🧭 유형 지도 · SHAP 요인 분석")
require_data()

kind_lab = st.radio("대상", ["투수", "타자"], horizontal=True)
kind = "pitcher" if kind_lab == "투수" else "batter"
cm = meta("clusters_meta").get(kind, {})
cl = load(f"{kind}_clusters")
if cl.empty:
    st.info("군집 결과가 없습니다.")
    st.stop()
seasons = sorted(cl["season"].unique(), reverse=True)
season = st.selectbox("시즌", ["전체"] + seasons)
d = cl if season == "전체" else cl[cl["season"] == season]
nm = names(kind)
d = d.assign(name=d[kind].map(nm))
types = sorted(cl["유형"].unique())
color = {t: CAT[i % 8] for i, t in enumerate(types)}

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    for t in types:
        x = d[d["유형"] == t]
        fig.add_trace(go.Scattergl(x=x["pc1"], y=x["pc2"], mode="markers", name=f"유형 {t} ({len(x)})",
                                 marker=dict(size=8, color=color[t], line=dict(color=SURFACE, width=1), opacity=.85),
                                 customdata=np.c_[x["name"], x["season"], x["gmm_확신도"]],
                                 hovertemplate="%{customdata[0]} (%{customdata[1]})<br>GMM 확신도 %{customdata[2]:.2f}<extra>유형 " + t + "</extra>"))
    pv = cm.get("pca_var", [0, 0])
    fig.update_layout(title=f"PCA 2차원 유형 지도 (설명력 {pv[0] * 100:.0f}% + {pv[1] * 100:.0f}%)", height=520,
                      xaxis_title="주성분 1", yaxis_title="주성분 2")
    st.plotly_chart(fig, width="stretch", config=PCFG)
with c2:
    st.markdown(f"##### k = {cm.get('k')} 선택 근거")
    ks = load(f"{kind}_k_selection")
    fig = go.Figure(go.Scatter(x=ks["k"], y=ks["silhouette"], mode="lines+markers", line=dict(color=CAT[0], width=2), marker=dict(size=8),
                               hovertemplate="k=%{x}: 실루엣 %{y:.3f}<extra></extra>"))
    fig.add_vline(x=cm.get("k"), line=dict(color=INK2, dash="dot"))
    fig.update_layout(height=200, title="실루엣 계수 (높을수록 잘 분리)", margin=dict(t=40, b=30), xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch", config=PCFG)
    show = ks.rename(columns={"inertia": "엘보우(관성)", "silhouette": "실루엣", "davies_bouldin": "Davies-Bouldin", "gmm_bic": "GMM BIC"})
    st.dataframe(show, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in show.columns if c != "k"})
    rm = cm.get("vif_removed", [])
    note("VIF > 10 제거 변수: " + (", ".join(f"{n}" for n, v in rm) or "없음") +
         f" · 부트스트랩 ARI {cm.get('bootstrap_ari_mean', float('nan')):.2f} (1에 가까울수록 안정)")

st.subheader("유형별 특징")
prof = load(f"{kind}_cluster_profile").set_index("유형")
raw = load(f"{kind}_cluster_raw").set_index("유형")
fig = go.Figure(go.Heatmap(z=prof.values, x=prof.columns, y=[f"유형 {t}" for t in prof.index], colorscale=DIV, zmid=0, zmin=-2, zmax=2,
                           text=np.vectorize(lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}")(raw.values.astype(float)), texttemplate="%{text}",
                           textfont=dict(size=11), xgap=2, ygap=2,
                           colorbar=dict(title="Z", thickness=10), hovertemplate="%{y} · %{x}<br>Z %{z:.2f}<extra></extra>"))
fig.update_layout(height=120 + 50 * len(prof), title="유형별 평균 (색 = 리그 대비 Z점수, 숫자 = 실제 평균)", xaxis=dict(tickangle=-35))
st.plotly_chart(fig, width="stretch", config=PCFG)
from scipy.stats import norm
rc1, rc2 = st.columns([3, 2])
with rc1:
    # 유형별 평균 Z점수 → 백분위(0~100)로 변환해 레이더로 비교
    rs = [(f"유형 {t}", list(norm.cdf(prof.loc[t].values.astype(float)) * 100), color[t]) for t in prof.index]
    st.plotly_chart(radar(rs, list(prof.columns), "유형별 레이더 (리그 평균 = 50)", height=460), width="stretch", config=PCFG)
with rc2:
    note("각 축은 유형 평균의 Z점수를 정규분포 백분위로 바꾼 값입니다. 50이 리그 평균, 바깥으로 갈수록 그 지표가 높습니다. "
         "(BB%·HardHit%처럼 낮을수록 좋은 지표도 '값이 높음' 방향 그대로 표시)")
for t in prof.index:
    st.markdown(f"<span class='badge' style='border-left:4px solid {color[t]}'>유형 {t}</span> {cm.get('desc', {}).get(t, '')} · {cm.get('sizes', {}).get(t, 0)}명·시즌", unsafe_allow_html=True)
pick = st.selectbox("유형 소속 선수 보기", types, format_func=lambda t: f"유형 {t}")
mem = d[d["유형"] == pick].sort_values("gmm_확신도", ascending=False)[["name", "season", "gmm_확신도"]]
st.dataframe(mem.rename(columns={"name": "선수", "season": "시즌", "gmm_확신도": "GMM 확신도"}), hide_index=True, height=240)

# ------------------------------------------------ SHAP
st.divider()
sm = meta("shap_meta")
if kind == "pitcher":
    st.subheader("SHAP: 무엇이 헛스윙을 만드나 (구위 모델)")
    sh = load("stuff_shap")
    info = sm.get("stuff", {})
    note(f"LightGBM으로 구종의 물리 특성 → 헛스윙률 예측. 투수 단위 5겹 교차검증 R² = {info.get('cv_r2', float('nan')):.3f}. "
         "SHAP = 각 특성이 예측을 평균에서 얼마나 밀어 올리거나 내렸는지.")
else:
    st.subheader("SHAP: 무엇이 타격 생산성(wOBA)을 만드나")
    sh = load("batter_shap")
    info = sm.get("batter", {})
    note(f"LightGBM으로 스윙·선구 특성 → wOBA 예측. 타자 단위 5겹 교차검증 R² = {info.get('cv_r2', float('nan')):.3f}.")
imp = pd.Series(info.get("importance", {})).sort_values()
c1, c2 = st.columns([2, 3])
with c1:
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="평균 |SHAP| (전역 중요도)", height=60 + 26 * len(imp), margin=dict(l=130))
    st.plotly_chart(fig, width="stretch", config=PCFG)
with c2:
    if len(imp):
        f = st.selectbox("특성별 SHAP 의존도", imp.index[::-1])
        x, y = sh[f"val_{f}"], sh[f"shap_{f}"]
        fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=6, color=CAT[0], opacity=.45),
                                   hovertemplate=f"{f} %{{x:.2f}}<br>SHAP %{{y:.4f}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.update_layout(title=f"{f} 값이 커질수록 예측이 어떻게 변하나", height=380, xaxis_title=f, yaxis_title="SHAP 값")
        st.plotly_chart(fig, width="stretch", config=PCFG)
