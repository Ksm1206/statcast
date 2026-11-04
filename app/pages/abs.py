import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import CAT, DIV, GRID, INK, INK2, SURFACE, grid_centers, load, meta, names, note, require_data, setup, zone_shapes

setup("🤖 ABS 챌린지 전후 비교 (2026)")
require_data()
am = meta("abs_meta")
its = load("abs_its")
if its.empty:
    st.info("ABS 분석 결과가 없습니다 (2026 시즌 데이터가 필요).")
    st.stop()

st.markdown("""
2026시즌부터 MLB는 **자동 볼·스트라이크(ABS) 챌린지**를 도입했습니다. 주심이 판정하되 팀당 경기마다 두 번 이의를 제기할 수 있습니다.
여기서는 **단절 시계열(ITS)** 로 '도입 효과'와 '원래 있던 추세·월별 계절성'을 분리하고, 판정 지도가 어디서 바뀌었는지, 어떤 투수가 영향을 받았는지 봅니다.
""")

st.subheader("단절 시계열 결과 (월 고정효과 + 연도 추세 + Newey-West HAC)")
show = its.copy()
st.dataframe(show, hide_index=True, column_config={
    "ABS 효과(%p)": st.column_config.NumberColumn(format="%+.2f"), "표준오차": st.column_config.NumberColumn(format="%.2f"),
    "p": st.column_config.NumberColumn(format="%.4f"), "q_FDR": st.column_config.NumberColumn(format="%.4f"),
    **{c: st.column_config.NumberColumn(format="%.2f") for c in show.columns if c not in ("지표", "p", "q_FDR", "추세 없는 p", "ABS 효과(%p)", "표준오차")},
    "추세 없는 p": st.column_config.NumberColumn(format="%.4f")})
note("추세를 넣은 모형과 뺀 모형의 결론이 같으면 효과가 견고합니다. 도입 후 시즌이 1개뿐이라 추세 추정에 민감할 수 있어 두 모형을 함께 봅니다. "
     "존 판정은 타자별로 전 시즌 공통 존(중앙값)을 써서, Statcast 존 정의가 시즌마다 바뀐 영향을 제거했습니다.")

m = load("abs_monthly_rates")
metric = st.selectbox("월별 추이 지표", [c for c in m.columns if c not in ("season", "month")], index=3)
fig = go.Figure()
seasons = sorted(m["season"].unique())
for i, s in enumerate(seasons):
    d = m[m["season"] == s]
    post = s >= am.get("abs_season", 2026)
    fig.add_trace(go.Scatter(x=d["month"], y=d[metric] * 100, mode="lines+markers", name=str(s),
                             line=dict(color=CAT[7] if post else ["#c3c2b7", "#a8a69e", "#898781", "#6e6c66", "#52514e"][i % 5],
                                       width=3 if post else 2), marker=dict(size=8 if post else 6)))
fig.update_layout(title=f"{metric} 월별 추이 (빨강 = ABS 시즌)", height=360, xaxis=dict(title="월", dtick=1), yaxis_title="%")
st.plotly_chart(fig, width="stretch")

st.subheader("판정 지도는 어디서 바뀌었나")
g = load("abs_grid_diff")
xc, zc = grid_centers()
c1, c2 = st.columns([3, 2])
with c1:
    Z = np.full((len(zc), len(xc)), np.nan)
    T = np.full(Z.shape, "", dtype=object)
    for _, r in g.iterrows():
        Z[int(r["gz"]), int(r["gx"])] = r["diff"] * 100
        T[int(r["gz"]), int(r["gx"])] = "*" if r["q_FDR"] < .05 else ""
    lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
    fig = go.Figure(go.Heatmap(x=xc, y=zc, z=Z, colorscale=DIV, zmid=0, zmin=-lim, zmax=lim, text=T, texttemplate="%{text}",
                               textfont=dict(color=INK, size=14), xgap=2, ygap=2, colorbar=dict(title="%p", thickness=10),
                               hovertemplate="x %{x:.2f}, 높이 %{y:.2f}<br>콜 스트라이크 변화 %{z:+.1f}%p<extra></extra>"))
    zone_shapes(fig)
    fig.update_layout(title="지켜본 공의 콜 스트라이크율 변화 (2026 − 도입 전, * = FDR 5% 유의)", height=460,
                      xaxis=dict(title="포수 시점 좌우(ft)", showgrid=False), yaxis=dict(title="높이(ft)", showgrid=False, scaleanchor="x"))
    st.plotly_chart(fig, width="stretch")
with c2:
    note(f"칸별로 도입 전과 2026의 콜 스트라이크 비율을 두 비율 z검정으로 비교하고 FDR로 보정했습니다. "
         f"유의하게 바뀐 칸: {am.get('cells_sig', 0)} / {am.get('cells', 0)}. "
         "파랑 = 스트라이크가 줄어든 곳, 빨강 = 늘어난 곳. 존 경계에서 변화가 몰리는지 보세요.")
    seas = load("abs_season")
    st.dataframe(seas.set_index("season").T.map(lambda v: f"{v * 100:.2f}%"))

st.subheader("누가 손해를 봤나 — 경계 콜 의존도 높은 투수")
did = load("abs_did_summary")
if len(did):
    st.dataframe(did, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in did.columns if c not in ("구간", "n")})
    note("ABS 직전 시즌에 존 경계 콜 스트라이크에 많이 의존한 투수일수록 볼넷이 늘었는지 회귀(HC1 강건 표준오차)로 확인합니다. "
         "같은 분석을 도입 전 구간(위약)에 적용해 원래부터 있던 추세가 아닌지 함께 봅니다.")
    dd = load("abs_pitcher_did")
    dd["name"] = dd["pitcher"].map(names("pitcher"))
    fig = go.Figure(go.Scatter(x=dd["edge_dep"] * 100, y=dd["ΔBB%"] * 100, mode="markers",
                               marker=dict(size=7, color=CAT[0], opacity=.7, line=dict(color=SURFACE, width=1)), text=dd["name"],
                               hovertemplate="%{text}<br>경계 의존도 %{x:.1f}%<br>ΔBB% %{y:+.2f}%p<extra></extra>"))
    b = np.polyfit(dd["edge_dep"] * 100, dd["ΔBB%"] * 100, 1)
    xs = np.linspace(dd["edge_dep"].min() * 100, dd["edge_dep"].max() * 100, 20)
    fig.add_trace(go.Scatter(x=xs, y=b[0] * xs + b[1], mode="lines", line=dict(color=INK, width=2), hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=INK2, width=1, dash="dot"))
    fig.update_layout(title="직전 시즌 경계 콜 의존도 vs 2026 볼넷률 변화", height=420, showlegend=False,
                      xaxis_title="콜 스트라이크 중 경계 공 비율 (%)", yaxis_title="BB% 변화 (%p)")
    st.plotly_chart(fig, width="stretch")
