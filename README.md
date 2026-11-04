# ⚾ 55만 개의 맞대결 — Statcast로 해부한 MLB 투수와 타자

2021–2026 MLB 전 투수·타자의 Statcast 투구 데이터를 수집해 **선수 해부 리포트 → 유형 분류 → 운 vs 실력 → 맞대결 예측 → ABS 전후 비교**까지 분석하고, Streamlit 앱으로 보여주는 프로젝트입니다.

> 저장소에 처음 들어 있는 `data/processed` 는 **테스트용 가짜 데이터**입니다(앱 상단에 경고 표시).
> 아래 2단계(GitHub Actions)를 한 번 실행하면 실제 데이터로 바뀌고 경고도 사라집니다.

---

## 실행 순서 (3단계)

### 1. GitHub에 올리기
이 폴더 전체를 새 저장소에 올립니다. (`data/raw`, `data/interim` 은 `.gitignore`로 제외됨)

### 2. GitHub Actions로 실제 데이터 만들기
1. 저장소 **Settings → Actions → General → Workflow permissions** 에서 **Read and write permissions** 선택 후 저장
2. **Actions** 탭 → **데이터 수집 및 분석** → **Run workflow**
   - `seasons`: `2021-2026` (처음엔 `2026`만 넣어 빠르게 확인해도 됨)
3. 끝나면 봇이 `data/processed` 를 자동 커밋합니다.

- 소요 시간: 시즌당 대략 10~40분 (Baseball Savant 서버 상황에 따라 다름). 최대 약 6시간 제한.
- 중간에 실패해도 받은 월 단위 파일은 캐시에 저장되어, 다시 실행하면 **이어서** 받습니다.
- 나중에 `2026`만 다시 받아도, 캐시에 있는 이전 시즌과 함께 전체를 다시 분석합니다.

### 3. Streamlit Community Cloud에 배포
1. https://share.streamlit.io → **Create app** → 이 저장소 선택
2. **Main file path**: `app/streamlit_app.py`
3. Deploy. (`requirements.txt` 는 자동 설치됨)
4. Actions가 데이터를 새로 커밋하면 앱도 자동으로 다시 시작됩니다.

---

## 로컬에서 실행 (선택)
```bash
pip install -r requirements-pipeline.txt
python scripts/run_all.py --seasons 2026          # 수집 → 집계 → 분석
python scripts/run_all.py --skip-collect          # 이미 받은 데이터로 다시 분석
streamlit run app/streamlit_app.py
```
가짜 데이터로 전체 흐름을 시험하려면:
```bash
MLB_RAW_DIR=/tmp/raw python tests/make_synthetic.py --seasons 2024-2026
MLB_RAW_DIR=/tmp/raw python scripts/run_all.py --skip-collect
python tests/test_app.py                          # 모든 페이지 오류 검사
```

### FanGraphs(fWAR, wRC+, Stuff+)
수집기가 pybaseball로 시도하지만 FanGraphs가 자동 접속을 막는 경우가 있습니다(실패해도 나머지는 정상 진행).
그럴 땐 FanGraphs 리더보드에서 CSV를 내려받아 아래 이름으로 `data/external/` 에 넣고 다시 분석하세요.
`fangraphs_pitching_2026.csv`, `fangraphs_batting_2026.csv` (CSV에 `MLBAMID` 컬럼이 있어야 매칭됨)

---

## 데이터
| 데이터 | 출처 | 단위 |
|---|---|---|
| 투구·타구 (구속, 회전, 무브먼트, 위치, 타구속도·발사각, 기대값, 배트 스피드) | Baseball Savant Statcast | 투구 1개 |
| ERA, 이닝, 승패, 홈런, 도루, 이름, 팀 | MLB Stats API | 선수·시즌 |
| 생일, 투타 | MLB Stats API | 선수 |
| fWAR, wRC+, Stuff+ (선택) | FanGraphs | 선수·시즌 |

- 정규시즌만 본 분석에 사용, 포스트시즌(`game_type` F/D/L/W)은 별도 집계
- 최소 표본: 투수 300구, 타자 150타석 (민감도 분석으로 기준을 바꿔 확인)
- 좌투수는 수평 무브먼트를 반전해 좌·우 투수를 같은 기준(팔 쪽 = +)으로 비교

## 분석 구성과 기법
| 단계 | 내용 | 기법 |
|---|---|---|
| 선수 해부 | 성적 카드, 리그 백분위, 구종 수·구사율(좌/우·카운트별), 구종별 물리 특성·위치·성적, 볼배합 | 집계, 백분위, 마르코프 전이행렬, 엔트로피 |
| 유형 분류 | 투수·타자 유형, 유사 선수 | VIF, 표준화, K-means, GMM, 실루엣·Davies-Bouldin·BIC, PCA, 부트스트랩 ARI, 코사인 유사도 |
| 요인 분석 | 구위 지수(물리 특성 → 헛스윙률), 타자 생산성 요인 | LightGBM, GroupKFold, TreeSHAP |
| 운 vs 실력 | wOBA−xwOBA, ERA 잔차, 연도 간 지속성 | Huber 로버스트 회귀, Ljung-Box, Newey-West HAC, Benjamini-Hochberg FDR |
| 공간 분석 | 스트라이크존 약점 구역 | Global Moran's I, LISA, 순열검정 |
| 맞대결 | 안 만난 쌍도 결과 확률 예측, 최적 타순 | Log5, 유형 교차표, 다항 로지스틱, LightGBM, 행렬 분해(ALS), 경험적 베이즈 축소, 보정 곡선 |
| ABS(2026) | 자동 볼판정 챌린지 효과 | 단절 시계열(월 고정효과 + HAC), 두 비율 z검정 + FDR, 이중차분형 비교, 위약 검정 |
| 시간 | 에이징 커브, 구속 급락 경보 | 혼합효과모형, 델타 방법, 변화점 탐지 |

맞대결 모델은 **시즌 s의 타석을 시즌 s−1의 특성으로 예측**해 누수를 막고, 2026시즌 전체로 검증합니다.

## 폴더 구조
```
app/                 Streamlit 앱 (메인: app/streamlit_app.py, 페이지: app/pages/)
src/mlbproj/         파이프라인
  collect.py         수집
  clean.py           정리·파생 변수
  aggregate.py       선수 해부 집계
  build.py           시즌별 실행·저장
  analysis/          군집, SHAP, 운, 공간, 맞대결, ABS, 에이징, 검증
scripts/run_all.py   전체 실행
tests/               가짜 데이터 생성기, 앱 페이지 테스트
data/processed/      앱이 읽는 결과 (Actions가 갱신)
.github/workflows/   자동 수집 워크플로
```

## 한계
- 맞대결 모델은 수비, 구장, 날씨, 피로도를 반영하지 않음
- 2026 ABS 도입으로 판정 기준이 바뀌어, 존 관련 지표의 시즌 간 비교는 주의
- 구종 라벨은 Statcast 분류를 그대로 사용
- 군집 이름은 자동 요약이며 해석은 분석자의 몫
