"""모든 페이지를 헤드리스로 실행해 예외가 없는지 확인: python tests/test_app.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
fails = 0
for page in sorted((APP / "pages").glob("*.py")):
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    errs = [e.value for e in at.exception]
    print(f"{'FAIL' if errs else 'ok  '} {page.name}")
    for e in errs:
        print("   ", str(e)[:2000])
    fails += bool(errs)
sys.exit(1 if fails else 0)
