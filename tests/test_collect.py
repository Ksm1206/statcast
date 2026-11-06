"""수집 실패 판정 확인 (네트워크 없이): python tests/test_collect.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlbproj import collect  # noqa: E402

collect.time.sleep = lambda s: None


def respond(text):
    collect.requests.get = lambda *a, **k: SimpleNamespace(text=text, raise_for_status=lambda: None)


# 경기 없는 날: 헤더만 있는 CSV → 빈 표 (정상)
respond('﻿"pitch_type","game_date"\n')
assert collect._fetch_day_direct(2025, collect.date(2025, 7, 16)).empty

# 차단·오류 페이지(HTML)나 빈 응답 → 빈 표로 넘기지 않고 실패
for bad in ("<html>Too Many Requests</html>", ""):
    respond(bad)
    try:
        collect._fetch_day_direct(2025, collect.date(2025, 7, 20))
        raise AssertionError(f"실패해야 함: {bad!r}")
    except RuntimeError:
        pass

print("ok   test_collect.py")
