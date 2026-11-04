# 실행 방법

## 1. GitHub Actions로 데이터 수집·분석
1. 저장소 **Settings → Actions → General → Workflow permissions**에서 **Read and write permissions** 선택 후 저장
2. **Actions** 탭 → **데이터 수집 및 분석** → **Run workflow**
   - `seasons`: `2021-2026` (처음엔 `2026`만 넣어 빠르게 확인해도 됨)
3. 끝나면 `data/processed`가 자동으로 커밋됩니다.

- 소요 시간: 시즌당 대략 10~40분 (Baseball Savant 서버 상황에 따라 다름), 최대 약 6시간
- 중간에 실패해도 받은 월 단위 파일은 캐시에 남아, 다시 실행하면 이어서 받습니다.
- 나중에 한 시즌만 다시 받아도, 캐시에 있는 이전 시즌과 함께 전체를 다시 분석합니다.

## 2. Streamlit Community Cloud 배포
1. https://share.streamlit.io → **Create app** → 이 저장소 선택
2. **Main file path**: `app/streamlit_app.py`
3. Deploy (`requirements.txt` 자동 설치)
4. Actions가 데이터를 새로 커밋하면 앱도 자동으로 다시 시작됩니다.

## 로컬 실행 (선택)
```bash
pip install -r requirements-pipeline.txt
python scripts/run_all.py --seasons 2026          # 수집 → 집계 → 분석
python scripts/run_all.py --skip-collect          # 받은 데이터로 다시 분석
streamlit run app/streamlit_app.py
```

테스트용 가짜 데이터로 전체 흐름 확인:
```bash
MLB_RAW_DIR=/tmp/raw python tests/make_synthetic.py --seasons 2024-2026
MLB_RAW_DIR=/tmp/raw python scripts/run_all.py --skip-collect
python tests/test_app.py
```

## FanGraphs 지표 (fWAR, wRC+, Stuff+)
자동 수집이 막히면 FanGraphs 리더보드에서 CSV를 받아 `data/external/`에 넣고 다시 분석합니다.
파일 이름: `fangraphs_pitching_2026.csv`, `fangraphs_batting_2026.csv` (`MLBAMID` 컬럼 필요)
