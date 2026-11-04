"""전체 파이프라인 실행.

    python scripts/run_all.py                  # 수집 → 집계 → 분석 전부
    python scripts/run_all.py --skip-collect   # 이미 data/raw 에 데이터가 있을 때
    python scripts/run_all.py --only analysis  # 분석만 다시
    python scripts/run_all.py --seasons 2026   # 특정 시즌만 수집
"""
from __future__ import annotations

import argparse
import time

import _path  # noqa: F401  (src 경로 추가)
from mlbproj import build, collect
from mlbproj.analysis import abs_effect, aging, clustering, luck, matchup, shap_model, spatial, validation

STEPS = [("clustering", clustering.main), ("shap", shap_model.main), ("luck", luck.main),
         ("spatial", spatial.main), ("matchup", matchup.main), ("abs", abs_effect.main),
         ("aging", aging.main), ("validation", validation.main)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--only", choices=["collect", "build", "analysis"], default=None)
    ap.add_argument("--skip-fangraphs", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 받은 시즌도 다시 수집")
    a = ap.parse_args()
    t0 = time.time()
    if a.only in (None, "collect") and not a.skip_collect:
        args = (["--seasons", a.seasons] if a.seasons else []) + (["--skip-fangraphs"] if a.skip_fangraphs else []) + (["--force"] if a.force else [])
        collect.main(args)
    if a.only in (None, "build"):
        build.main()
    if a.only in (None, "analysis"):
        for name, fn in STEPS:
            t = time.time()
            fn()
            print(f"  ({name} {time.time() - t:.0f}초)")
    print(f"완료: {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
