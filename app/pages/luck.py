import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import PCFG, CAT, INK, SURFACE, load, meta, names, note, require_data, setup

setup("📈 기대 성적 — 결과는 과정을 따라오는가")
require_data()
lm = meta("luck_meta")

st.markdown("""
**질문:** 올해 성적이 타구의 질(과정)에 비해 지나치게 좋거나 나빴던 선수는 누구이고, 그 차이는 다음 시즌까지 이어지는가?
**방법:** 타석마다 *실제 결과 가치(wOBA)* 와 *타구 속도·발사각으로 본 기대 가치(xwOBA)* 의 차이를 평균냅니다.
경기 순서에 따른 자기상관을 **Ljung-Box**로 확인하고 **Newey-West(HAC)** 표준오차로 보정, 수백 명을 동시에 검정하므로 **Benjamini-Hochberg FDR**로 다시 보정합니다.
""")
if lm:
    c = st.columns(4)
    c[0].metric("유의한 투수 (보정 전 → 후)", f"{lm['pitcher_naive_sig']} → {lm['pitcher_sig']}", f"/ {lm['pitcher_n']}명·시즌", delta_color="off")
    c[1].metric("유의한 타자 (보정 전 → 후)", f"{lm['batter_naive_sig']} → {lm['batter_sig']}", f"/ {lm['batter_n']}명·시즌", delta_color="off")
    if lm.get("luck_persistence_b") is not None:
        c[2].metric("갭의 연도 간 상관 (타자)", f"{lm['luck_persistence_b']:.2f}", "낮을수록 일시적", delta_color="off")
    if lm.get("skill_persistence_b_K") is not None:
        c[3].metric("K% 연도 간 상관 (실력 비교용)", f"{lm['skill_persistence_b_K']:.2f}", delta_color="off")

kind = st.radio("대상", ["타자", "투수"], horizontal=True)
key = "batter" if kind == "타자" else "pitcher"
t = load(f"luck_{key}")
season = st.selectbox("시즌", sorted(t["season"].unique(), reverse=True))
d = t[t["season"] == season].copy()
d["name"] = d[key].map(names(key))

fig = go.Figure()
ns, sg = d[~d["유의(FDR 5%)"]], d[d["유의(FDR 5%)"]]
fig.add_trace(go.Scattergl(x=ns["xwOBA"], y=ns["wOBA"], mode="markers", name="유의하지 않음",
                         marker=dict(size=7, color="#c3c2b7", line=dict(color=SURFACE, width=1)),
                         customdata=np.c_[ns["name"], ns["PA"], ns["q_FDR"]],
                         hovertemplate="%{customdata[0]}<br>xwOBA %{x:.3f} → wOBA %{y:.3f}<br>%{customdata[1]}타석 · q=%{customdata[2]:.3f}<extra></extra>"))
fig.add_trace(go.Scatter(x=sg["xwOBA"], y=sg["wOBA"], mode="markers+text", name="FDR 5% 유의", text=sg["name"],
                         textposition="top center", textfont=dict(size=10, color=INK),
                         marker=dict(size=10, color=CAT[7], line=dict(color=SURFACE, width=2)),
                         customdata=np.c_[sg["name"], sg["PA"], sg["q_FDR"]],
                         hovertemplate="%{customdata[0]}<br>xwOBA %{x:.3f} → wOBA %{y:.3f}<br>%{customdata[1]}타석 · q=%{customdata[2]:.3f}<extra></extra>"))
lo, hi = float(d[["xwOBA", "wOBA"]].min().min()), float(d[["xwOBA", "wOBA"]].max().max())
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK, dash="dot", width=1), showlegend=False, hoverinfo="skip"))
side = "허용 " if key == "pitcher" else ""
fig.update_layout(title=f"{side}xwOBA(기대) vs {side}wOBA(실제) — 점선 위 = 기대보다 결과가 좋음(타자)/나쁨(투수)", height=520,
                  xaxis_title="xwOBA", yaxis_title="wOBA")
st.plotly_chart(fig, width="stretch", config=PCFG)

c1, c2 = st.columns(2)
cols = ["name", "PA", "wOBA", "xwOBA", "차이", "z_HAC", "q_FDR"]
cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in ("wOBA", "xwOBA", "차이", "q_FDR")}
cfg["z_HAC"] = st.column_config.NumberColumn(format="%.2f")
if key == "batter":
    c1.markdown("##### 다음 시즌 반등 후보 (실제 < 기대)")
    c1.dataframe(d.sort_values("차이").head(15)[cols].rename(columns={"name": "선수"}), hide_index=True, column_config=cfg)
    c2.markdown("##### 하락 주의 (실제 > 기대)")
    c2.dataframe(d.sort_values("차이", ascending=False).head(15)[cols].rename(columns={"name": "선수"}), hide_index=True, column_config=cfg)
else:
    c1.markdown("##### 다음 시즌 반등 후보 (허용 결과가 기대보다 나빴음)")
    c1.dataframe(d.sort_values("차이", ascending=False).head(15)[cols].rename(columns={"name": "선수"}), hide_index=True, column_config=cfg)
    c2.markdown("##### 하락 주의 (허용 결과가 기대보다 좋았음)")
    c2.dataframe(d.sort_values("차이").head(15)[cols].rename(columns={"name": "선수"}), hide_index=True, column_config=cfg)

if key == "pitcher":
    st.subheader("ERA는 실력을 얼마나 반영하나 — Huber 로버스트 회귀")
    e = load("era_residuals")
    e = e[e["season"] == season].copy()
    e["name"] = e["pitcher"].map(names("pitcher"))
    fig = go.Figure(go.Scattergl(x=e["기대 ERA"], y=e["ERA"], mode="markers", marker=dict(size=7, color=CAT[0], opacity=.7, line=dict(color=SURFACE, width=1)),
                               customdata=np.c_[e["name"], e["IP"]],
                               hovertemplate="%{customdata[0]} (%{customdata[1]:.0f}이닝)<br>기대 %{x:.2f} · 실제 %{y:.2f}<extra></extra>"))
    m = float(min(e["기대 ERA"].min(), e["ERA"].min())), float(max(e["기대 ERA"].max(), e["ERA"].max()))
    fig.add_trace(go.Scatter(x=m, y=m, mode="lines", line=dict(color=INK, dash="dot", width=1), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=430, xaxis_title="기대 ERA (K%·BB%·GB%·HardHit%)", yaxis_title="실제 ERA", showlegend=False)
    st.plotly_chart(fig, width="stretch", config=PCFG)
    if lm.get("huber_coef"):
        note(f"표준화 계수(Huber): {lm['huber_coef']} · OLS와 잔차 순위 상관 ρ = {lm['huber_vs_ols_spearman']:.3f} "
             "(이상치 몇 명이 기준선을 끌고 가지 않음을 확인). ERA 잔차의 연도 간 상관 "
             f"{lm.get('era_resid_persistence') or float('nan'):.2f} vs K% {lm.get('k_persistence') or float('nan'):.2f}.")
