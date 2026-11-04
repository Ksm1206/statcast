import pandas as pd
import streamlit as st

from lib import load, meta, note, require_data, setup

setup("⚾ 55만 개의 맞대결 — Statcast로 해부한 MLB")
require_data()

ps, bs = load("pitcher_season"), load("batter_season")
mm = meta("matchup_meta")
seasons = sorted(ps["season"].unique())

c = st.columns(5)
c[0].metric("분석 기간", f"{seasons[0]}–{seasons[-1]}")
c[1].metric("투구 수", f"{int(ps['pitches'].sum()):,}")
c[2].metric("투수 (선수·시즌)", f"{len(ps):,}")
c[3].metric("타자 (선수·시즌)", f"{len(bs):,}")
c[4].metric("실제 맞대결 쌍", f"{mm.get('observed_pairs', 0):,}")

st.markdown("""
### 이 앱으로 할 수 있는 것
| 메뉴 | 내용 | 사용 기법 |
|---|---|---|
| **투수 리포트** | 성적 카드, 구종 수·구사율, 구종별 무브먼트·위치·성적, 카운트별 볼배합, 구종 전이 | 집계, 백분위, 마르코프 전이행렬 |
| **타자 리포트** | 타구 질, 선구안, 구종·존·구속대·카운트별 성적, 스프레이 차트, 약점 존 | 존 격자, LISA 핫스팟 |
| **유형 지도 · SHAP** | 투수·타자 유형 분류, 유사 선수, 성적을 만드는 요인 | VIF, K-means, GMM, PCA, LightGBM, SHAP |
| **맞대결 시뮬레이터** | 한 번도 안 만난 투수·타자도 결과 확률 예측, 상대 선발 기준 최적 타순 | Log5, 다항 로지스틱, LightGBM, 행렬 분해(ALS) |
| **기대 성적** | 실제 성적(wOBA)과 타구 질로 본 기대 성적(xwOBA)의 차이, 다음 시즌 반등·하락 후보 | Huber 회귀, Newey-West HAC, Ljung-Box, FDR |
| **ABS 전후 비교** | 2026 자동 볼판정 챌린지 도입이 바꾼 것 | 단절 시계열(ITS), 이중차분형 비교, 위약 검정 |
| **검증 · 방법론** | 위협 → 검증 → 결과 표, 모델 비교, 에이징 커브 | 혼합효과모형, 델타 방법, 변화점 탐지 |
""")

st.subheader("왜 1:1 기록이 아니라 '특성'으로 예측하나")
if mm:
    a, b = mm["possible_pairs_per_season"], mm["pa_per_season"]
    st.markdown(f"한 시즌에 가능한 투수·타자 조합은 약 **{a:,}쌍**인데 실제 타석은 약 **{b:,.0f}번**뿐입니다. "
                f"쌍당 평균 **{b / a:.2f}타석**이라 대부분의 조합은 만난 적이 없고, 20타석 이상 만난 쌍은 "
                f"**{mm['pairs_ge_20pa']:,}쌍**에 불과합니다. 그래서 투수와 타자 각각의 수천 구 데이터로 만든 특성으로 맞대결을 예측합니다.")
note("데이터: Baseball Savant(Statcast), MLB Stats API, FanGraphs(선택). 정규시즌 기준, 포스트시즌은 별도 집계.")
