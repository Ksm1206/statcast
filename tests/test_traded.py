"""이적 선수 시즌 성적 정리 확인: python tests/test_traded.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlbproj.collect import _dedupe_traded  # noqa: E402


def row(pid, team, g, ab, h, bb, hbp, sf, tb, avg):
    return {"player_id": pid, "name": "x", "team": team, "gamesPlayed": g, "atBats": ab, "hits": h, "baseOnBalls": bb,
            "hitByPitch": hbp, "sacFlies": sf, "totalBases": tb, "avg": avg, "obp": .3, "slg": .4, "ops": .7}


def one(rows):
    out = _dedupe_traded(pd.DataFrame(rows), "gamesPlayed")
    assert len(out) == 1, out
    return out.iloc[0]


# 팀별 행만 있음 (경기 수 같음): 둘 다 합산하고 비율은 합계로 다시 계산
r = one([row(1, "BOS", 10, 40, 10, 4, 1, 0, 15, .250), row(1, "SF", 10, 40, 12, 2, 0, 1, 20, .300)])
assert r["gamesPlayed"] == 20 and r["hits"] == 22 and r["team"] == "BOS/SF"
assert abs(r["avg"] - 22 / 80) < 1e-9, r["avg"]
assert abs(r["obp"] - (22 + 6 + 1) / (80 + 6 + 1 + 1)) < 1e-9
assert abs(r["slg"] - 35 / 80) < 1e-9 and abs(r["ops"] - (r["obp"] + r["slg"])) < 1e-9

# 팀별 행만 있음 (작은 두 팀 경기 수 합 = 큰 팀): 합계 행으로 착각하지 않음
r = one([row(2, "A", 20, 80, 20, 0, 0, 0, 20, .250), row(2, "B", 10, 40, 10, 0, 0, 0, 10, .250),
         row(2, "C", 10, 40, 10, 0, 0, 0, 10, .250)])
assert r["gamesPlayed"] == 40 and r["atBats"] == 160

# 합계 행(팀 없음)이 있으면 그 행을 그대로 쓰고 팀 이름만 채움
r = one([row(3, "", 30, 120, 36, 0, 0, 0, 50, .300), row(3, "NYY", 20, 80, 24, 0, 0, 0, 30, .300),
         row(3, "TB", 10, 40, 12, 0, 0, 0, 20, .300)])
assert r["gamesPlayed"] == 30 and r["avg"] == .300 and r["team"] == "NYY/TB"

# 한 팀 선수는 그대로
r = one([row(4, "LAD", 50, 200, 50, 0, 0, 0, 80, .250)])
assert r["avg"] == .250 and r["team"] == "LAD"

print("ok   test_traded.py")
