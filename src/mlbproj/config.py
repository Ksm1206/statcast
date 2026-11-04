"""프로젝트 전역 설정: 경로, 시즌 범위, 표본 기준, 구종 그룹."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = Path(os.environ.get("MLB_RAW_DIR", DATA / "raw"))  # Statcast 원본 (git에 올리지 않음)
EXTERNAL = DATA / "external"  # FanGraphs CSV 등 직접 받은 파일 (선택)
INTERIM = Path(os.environ.get("MLB_INTERIM_DIR", DATA / "interim"))  # 파이프라인 전용 중간 결과 (git 제외)
PROCESSED = Path(os.environ.get("MLB_PROCESSED_DIR", DATA / "processed"))  # 앱이 읽는 집계 결과

for _p in (RAW, EXTERNAL, INTERIM, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

_s = os.environ.get("MLB_SEASONS", "2021-2026")  # 테스트용으로 바꿀 수 있음
SEASONS = list(range(int(_s.split("-")[0]), int(_s.split("-")[-1]) + 1))
ABS_SEASON = 2026                        # ABS 챌린지 도입 시즌
TEST_SEASON = 2026                       # 맞대결 모델 검증 시즌

# 최소 표본 기준 (민감도 분석에서 바꿔가며 확인)
MIN_PITCHES_PITCHER = 300
MIN_PA_BATTER = 150
MIN_PITCHES_PER_TYPE = 30   # 구종별 성적을 보여줄 최소 투구 수

# Statcast 원본에서 유지할 컬럼 (없는 컬럼은 NaN으로 채움)
KEEP_COLS = [
    "game_date", "game_year", "game_type", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter", "player_name", "stand", "p_throws", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    "release_pos_x", "release_pos_z", "release_extension",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "type", "description", "events", "bb_type",
    "launch_speed", "launch_angle", "launch_speed_angle", "hc_x", "hc_y",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "delta_run_exp", "bat_speed", "swing_length",
    "n_thruorder_pitcher", "age_pit", "age_bat",
]

# 구종 → 그룹 (타자 분석, 맞대결 모델에서 사용)
PITCH_GROUP = {
    "FF": "패스트볼", "SI": "패스트볼", "FC": "패스트볼", "FA": "패스트볼",
    "SL": "브레이킹", "ST": "브레이킹", "CU": "브레이킹", "KC": "브레이킹",
    "SV": "브레이킹", "CS": "브레이킹", "SC": "브레이킹",
    "CH": "오프스피드", "FS": "오프스피드", "FO": "오프스피드",
    "KN": "기타", "EP": "기타", "PO": "기타",
}

PITCH_NAME_KO = {
    "FF": "포심", "SI": "싱커", "FC": "커터", "SL": "슬라이더", "ST": "스위퍼",
    "CU": "커브", "KC": "너클커브", "SV": "슬러브", "CS": "슬로커브",
    "CH": "체인지업", "FS": "스플리터", "FO": "포크", "SC": "스크루볼",
    "KN": "너클볼", "EP": "이퓨스", "FA": "패스트볼",
}

# 히트맵 격자 (포수 시점, 단위 ft)
GRID_X = (-2.0, 2.0)
GRID_Z = (0.5, 4.5)
GRID_N = 10
