import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import CAT, INK, PCFG, SURFACE, filter_team, load, meta, note, require_data, setup, team_list

setup("🏆 종합 지수")
require_data()
cm = meta("composite_meta")

st.markdown("""
여러 지표를 하나의 **0~100점 종합 지수**로 합칩니다. (REVI 지수와 같은 설계 순서)
1. 지표 선정 → 2. 방향 통일 (낮을수록 좋은 지표는 뒤집음) → 3. 시즌별 Min-Max 정규화
→ 4. 가중치 계산 → 5. 가중합 → 6. 시즌별 Min-Max로 0~100점

가중치는 두 가지 방법으로 계산해 비교합니다.
- **엔트로피 가중법:** 선수들 사이에 값이 고르게 퍼져 있지 않고 차이가 큰 지표일수록(정보량이 많을수록) 가중치가 큼. 데이터의 분포만 봅니다.
- **SHAP 가중치:** LightGBM으로 실점 가치(투수 RV/100, 타자 wOBA)를 예측했을 때 각 지표의 평균 |SHAP| 비율. 성적에 대한 영향력을 봅니다.

**최종 지수는 다음 시즌 성적을 더 잘 예측한 쪽**으로 고릅니다.
""")

kind_lab = st.radio("대상", ["투수", "타자"], horizontal=True)
kind = "pitcher" if kind_lab == "투수" else "batter"
c = cm.get(kind, {})
W = load(f"composite_weights_{kind}")
V = load(f"composite_validation_{kind}")
S = load(f"composite_{kind}")
if S.empty:
    st.info("종합 지수 결과가 없습니다.")
    st.stop()

k = st.columns(3)
k[0].metric("최종 채택 지수", c.get("final", "–"))
k[1].metric("두 지수의 순위 일치도 (Spearman)", f"{c.get('entropy_vs_shap_spearman', float('nan')):.2f}")
k[2].metric("가중치 1위 (엔트로피 / SHAP)", f"{c.get('top_entropy_weight', '')} / {c.get('top_shap_weight', '')}")

c1, c2 = st.columns([3, 2])
with c1:
    w = W.sort_values("SHAP 가중치")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=w["지표"], x=w["엔트로피 가중치"] * 100, name="엔트로피 가중법", orientation="h",
                         marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                         text=[f"{v * 100:.0f}" for v in w["엔트로피 가중치"]], textposition="outside"))
    fig.add_trace(go.Bar(y=w["지표"], x=w["SHAP 가중치"] * 100, name="SHAP 가중치", orientation="h",
                         marker=dict(color=CAT[1], line=dict(color=SURFACE, width=2)),
                         text=[f"{v * 100:.0f}" for v in w["SHAP 가중치"]], textposition="outside"))
    fig.update_layout(barmode="group", title="지표별 가중치 (%)", height=80 + 46 * len(w), xaxis_title="가중치 (%)", margin=dict(l=110))
    st.plotly_chart(fig, width="stretch", config=PCFG)
with c2:
    st.markdown("##### 지수 검증 (순위 상관, 높을수록 좋음)")
    st.dataframe(V, hide_index=True, column_config={col: st.column_config.NumberColumn(format="%.3f") for col in V.columns if col != "지수"})
    note("같은 시즌: 이미 쓰는 성적 지표와 얼마나 맞는지 (방향 통일). 다음 시즌: 올해 지수가 내년 실점 가치를 얼마나 예측하는지. "
         "비교용으로 xwOBA 하나만 썼을 때의 예측력도 넣었습니다. 종합 지수가 이보다 높아야 '합친 의미'가 있습니다.")
    st.dataframe(W[["지표", "방향"]], hide_index=True)

st.subheader("두 지수는 같은 선수를 높게 평가하나")
season = st.selectbox("시즌", sorted(S["season"].unique(), reverse=True))
d = S[S["season"] == season].copy()
fig = go.Figure(go.Scattergl(x=d["엔트로피 지수"], y=d["SHAP 지수"], mode="markers", text=d["name"],
                             marker=dict(size=7, color=CAT[0], opacity=.6, line=dict(color=SURFACE, width=1)),
                             hovertemplate="%{text}<br>엔트로피 %{x:.2f} · SHAP %{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=INK, dash="dot", width=1), hoverinfo="skip"))
fig.update_layout(height=430, showlegend=False, xaxis_title="엔트로피 지수", yaxis_title="SHAP 지수",
                  title="점선에서 멀수록 두 방법의 평가가 다른 선수")
st.plotly_chart(fig, width="stretch", config=PCFG)

st.subheader(f"{season} 종합 지수 순위")
ss = load(f"{kind}_season")
d = d.merge(ss[[kind, "season", "팀"]], on=[kind, "season"], how="left")
teams = team_list(d)
team = st.selectbox("구단 (선택하지 않으면 전체)", teams, index=None, placeholder="전체 구단")
if team:
    d = filter_team(d, team)
d = d.sort_values("종합 지수", ascending=False).reset_index(drop=True)
d.index += 1
st.dataframe(d[["name", "팀", "종합 지수", "엔트로피 지수", "SHAP 지수", "시즌 내 백분위"]].rename(columns={"name": "선수"}),
             column_config={"종합 지수": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                            "엔트로피 지수": st.column_config.NumberColumn(format="%.2f"),
                            "SHAP 지수": st.column_config.NumberColumn(format="%.2f"),
                            "시즌 내 백분위": st.column_config.NumberColumn(format="%.2f")}, height=520)
note("순위는 종합 지수(소수점 둘째 자리까지 표시) 기준입니다. 표시값까지 같으면 반올림 전 값으로 순위를 정합니다.")
note("엔트로피 가중법은 분포만 보기 때문에 성적과 무관하게 선수 간 편차가 큰 지표에 가중치가 몰릴 수 있습니다. "
     "그래서 성적 영향력을 보는 SHAP 가중치와 함께 비교하고, 다음 시즌 예측력으로 최종 지수를 정했습니다.")
