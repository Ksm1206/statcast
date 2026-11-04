import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import CAT, INK, SURFACE, load, meta, names, note, require_data, setup

setup("🔬 검증 · 방법론")
require_data()

st.subheader("모형 검증 종합표")
v = load("validation_table")
st.dataframe(v, hide_index=True, width="stretch", height=min(40 + 36 * len(v), 800))

st.subheader("맞대결 모델 비교 (검증 시즌 전체)")
mm = meta("matchup_meta")
c = load("matchup_model_comparison")
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(c, hide_index=True, column_config={
        "로그 손실": st.column_config.NumberColumn(format="%.4f"), "Brier": st.column_config.NumberColumn(format="%.4f"),
        "개선율(vs 리그 평균)": st.column_config.NumberColumn(format="%.2%"), "xwOBA RMSE": st.column_config.NumberColumn(format="%.4f")})
    note(f"학습 {mm.get('train_pa', 0):,}타석 → 검증 {mm.get('test_pa', 0):,}타석 ({mm.get('test_season')}). "
         "타석 하나의 결과는 원래 우연이 매우 커서 개선율 자체는 작게 나오는 것이 정상입니다. 비교의 기준은 Log5보다 나은가입니다. "
         + (f"8타석 이상 만난 쌍 {mm['frequent_pairs_n']}개에서 예측-실제 상관 r = {mm['frequent_pairs_corr']:.2f}." if mm.get("frequent_pairs_corr") else ""))
with c2:
    cal = load("matchup_calibration")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["예측"], y=cal["실제"], mode="lines+markers", line=dict(color=CAT[0], width=2), marker=dict(size=8),
                             hovertemplate="예측 %{x:.3f} → 실제 %{y:.3f}<extra></extra>"))
    lo, hi = float(cal[["예측", "실제"]].min().min()), float(cal[["예측", "실제"]].max().max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK, dash="dot", width=1), hoverinfo="skip"))
    fig.update_layout(title="보정 곡선 (예측 10분위)", height=320, showlegend=False, xaxis_title="예측 기대 wOBA", yaxis_title="실제 xwOBA")
    st.plotly_chart(fig, width="stretch")
imp = load("matchup_importance").head(15).iloc[::-1]
fig = go.Figure(go.Bar(x=imp["중요도(gain)"], y=imp["특성"], orientation="h", marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2))))
fig.update_layout(title="맞대결 LightGBM 특성 중요도 Top 15 (p_ = 투수, b_ = 타자)", height=480, margin=dict(l=170))
st.plotly_chart(fig, width="stretch")

st.subheader("에이징 커브")
ag = meta("aging_meta")
c1, c2 = st.columns(2)
for col, name, title, unit in ((c1, "aging_pitcher_velo", "투수 패스트볼 구속", "mph"), (c2, "aging_batter_xwoba", "타자 xwOBA", "xwOBA")):
    d = load(name)
    if d.empty:
        continue
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["age"], y=d["혼합효과모형"], name="혼합효과모형", mode="lines", line=dict(color=CAT[0], width=2)))
    fig.add_trace(go.Scatter(x=d["age"], y=d["델타 방법"], name="델타 방법", mode="lines", line=dict(color=CAT[1], width=2, dash="dash")))
    fig.update_layout(title=title, height=340, xaxis_title="나이", yaxis_title=unit)
    col.plotly_chart(fig, width="stretch")
note("혼합효과모형: 선수마다 다른 출발점(무작위 절편)을 허용한 2차 곡선. 델타 방법: 같은 선수의 연속 시즌 변화량을 나이별로 누적. "
     "두 방법이 비슷하면 '잘하는 선수만 오래 남는' 생존 편향의 영향이 작다는 뜻입니다.")

st.subheader("시즌 중 구속 급락 경보 (변화점 탐지)")
cp = load("velo_changepoints")
a = cp[cp["구속 급락 경보"]].copy()
a["투수"] = a["pitcher"].map(names("pitcher"))
st.dataframe(a.sort_values("변화량")[["투수", "season", "변화 시점", "이전 평균", "이후 평균", "변화량", "t", "경기 수"]], hide_index=True,
             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("이전 평균", "이후 평균", "변화량", "t")})
note("경기별 패스트볼 평균 구속에서 평균이 가장 크게 바뀌는 지점을 찾고(이분 분할 t 통계량), −1mph 이상 하락 & t ≤ −3이면 경보.")

st.subheader("데이터 출처와 한계")
st.markdown("""
- **Statcast** (Baseball Savant): 투구 단위. 정규시즌과 포스트시즌을 `game_type`으로 분리.
- **MLB Stats API**: ERA, 이닝, 홈런, 도루 등 시즌 기록과 선수 이름·생일. FIP는 리그 상수로 직접 계산.
- **FanGraphs** (선택): fWAR, wRC+, Stuff+. 자동 수집이 막히면 리더보드 CSV를 `data/external`에 넣으면 됨.
- 한계: 좌표·구종 라벨은 Statcast 분류를 그대로 사용 / 2026 ABS로 판정 기준이 바뀌어 존 관련 지표는 시즌 간 비교에 주의 /
  맞대결 모델은 수비·구장·날씨를 반영하지 않음 / 군집 이름은 자동 요약이며 해석은 분석자 몫.
""")
