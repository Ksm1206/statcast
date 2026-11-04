"""앱 공통: 데이터 로딩(캐시), 색상 규칙, Plotly 템플릿, 공통 차트."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("MLB_PROCESSED_DIR", ROOT / "data" / "processed"))

from mlbproj.config import GRID_N, GRID_X, GRID_Z, PITCH_NAME_KO  # noqa: E402

# ---------------------------------------------------------------- 색상 (검증된 범주형 8색, 순서 고정)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 구종 색은 구종(개체)을 따라감: 어떤 투수 화면이든 포심은 항상 파랑
PITCH_COLOR = {"FF": CAT[0], "SL": CAT[1], "CH": CAT[2], "CU": CAT[3], "SI": CAT[4], "FC": CAT[5], "ST": CAT[6],
               "FS": CAT[7], "KC": "#a8a69e", "SV": "#a8a69e", "KN": "#a8a69e", "SC": "#a8a69e", "EP": "#a8a69e"}
GROUP_COLOR = {"패스트볼": CAT[0], "브레이킹": CAT[1], "오프스피드": CAT[2], "기타": MUTED}
SEQ = [[0, "#f3f7fd"], [0.15, "#cde2fb"], [0.35, "#86b6ef"], [0.55, "#3987e5"], [0.75, "#256abf"], [1, "#0d366b"]]
DIV = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1, "#b8302f"]]

pio.templates["mlb"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif", color=INK, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=CAT,
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK2), title_font=dict(color=INK2)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK2), title_font=dict(color=INK2)),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(color=INK)),
    legend=dict(font=dict(color=INK2), orientation="h", y=-0.18),
    margin=dict(l=50, r=20, t=40, b=40), title=dict(font=dict(size=15, color=INK)),
))
pio.templates.default = "mlb"

CSS = """
<style>
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stMetricLabel"] p {color:#52514e;}
.small-note {color:#52514e; font-size:0.85rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:6px; background:#f0efec; color:#0b0b0b;
        font-size:0.8rem; margin-right:4px; border:1px solid rgba(11,11,11,.1)}
</style>
"""


def setup(title: str):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if is_demo():
        st.warning("현재 표시되는 값은 **파이프라인 테스트용 가짜 데이터**입니다. GitHub Actions로 실제 데이터를 수집하면 자동으로 바뀝니다.", icon="⚠️")


# ---------------------------------------------------------------- 데이터
@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    f = DATA / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def meta(name: str) -> dict:
    f = DATA / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def is_demo() -> bool:
    return (DATA / "DEMO").exists()


def has_data() -> bool:
    return (DATA / "pitcher_season.parquet").exists()


def require_data():
    if not has_data():
        st.error("data/processed 폴더에 결과 파일이 없습니다. README의 'GitHub Actions로 데이터 만들기'를 먼저 실행하세요.")
        st.stop()


def ko(pt: str) -> str:
    return PITCH_NAME_KO.get(pt, pt)


def names(kind: str) -> dict[int, str]:
    t = load(f"{kind}_season")
    return t.sort_values("season").groupby(kind)["name"].last().to_dict()


def pick_player(kind: str, key: str, label: str | None = None, default_top: bool = True):
    """시즌 + 선수 선택. 반환: (player_id, season, row)."""
    t = load(f"{kind}_season")
    seasons = sorted(t["season"].unique(), reverse=True)
    c1, c2 = st.columns([1, 3])
    season = c1.selectbox("시즌", seasons, key=f"{key}_season")
    sub = t[t["season"] == season].sort_values("pitches" if kind == "pitcher" else "PA", ascending=False)
    sub = sub[sub["qualified"]] if c1.checkbox("규정 표본만", True, key=f"{key}_q") else sub
    opts = sub[kind].tolist()
    lab = {r[kind]: f"{r['name']}  ({r.get('팀', '') or ''})" for _, r in sub.iterrows()}
    pid = c2.selectbox(label or ("투수" if kind == "pitcher" else "타자"), opts, format_func=lambda i: lab.get(i, str(i)), key=f"{key}_pid")
    return pid, season, sub[sub[kind] == pid].iloc[0]


def fmt(v, kind="pct", d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "–"
    if kind == "pct":
        return f"{v * 100:.{d}f}%"
    if kind == "rate":
        return f"{v:.3f}".lstrip("0") if 0 <= v < 1 else f"{v:.3f}"
    return f"{v:.{d}f}"


# ---------------------------------------------------------------- 공통 차트
def percentile_bars(row: pd.Series, metrics: list[tuple[str, str, str]], title="리그 백분위 (규정 표본 기준)"):
    """Baseball Savant 스타일 백분위 막대. metrics: (컬럼, 표시명, 값 형식)."""
    labs, pcts, txt, cols = [], [], [], []
    for m, name, k in metrics:
        p = row.get(f"pct_{m}")
        if p is None or pd.isna(p):
            continue
        labs.append(name)
        pcts.append(p)
        txt.append(fmt(row.get(m), k))
        t = p / 100
        cols.append(_interp_div(t))
    fig = go.Figure(go.Bar(x=pcts, y=labs, orientation="h", marker=dict(color=cols, line=dict(color=SURFACE, width=2)),
                           text=[f"{int(round(p))}" for p in pcts], textposition="outside", textfont=dict(color=INK),
                           customdata=txt, hovertemplate="%{y}: %{customdata}<br>백분위 %{x:.0f}<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=AXIS, dash="dot", width=1))
    fig.update_layout(title=title, height=32 * len(labs) + 90, xaxis=dict(range=[0, 108], title="백분위 (높을수록 좋음)"),
                      yaxis=dict(autorange="reversed"), margin=dict(l=110), showlegend=False)
    return fig


def _interp_div(t: float) -> str:
    stops = [(0, (28, 92, 171)), (0.5, (192, 191, 186)), (1, (184, 48, 47))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            c = [int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3)]
            return f"rgb({c[0]},{c[1]},{c[2]})"
    return "rgb(184,48,47)"


def zone_shapes(fig, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.6, y1=3.4, line=dict(color=INK, width=2), **kw)
    for x in (-0.83 / 3 * 1, 0.83 / 3 * 1):
        fig.add_shape(type="line", x0=x, x1=x, y0=1.6, y1=3.4, line=dict(color=AXIS, width=1, dash="dot"), **kw)
    for y in (2.2, 2.8):
        fig.add_shape(type="line", x0=-0.83, x1=0.83, y0=y, y1=y, line=dict(color=AXIS, width=1, dash="dot"), **kw)
    fig.add_shape(type="path", path="M -0.71 0.25 L 0.71 0.25 L 0.71 0.1 L 0 -0.05 L -0.71 0.1 Z",
                  line=dict(color=AXIS, width=1), fillcolor="#f0efec", **kw)


def grid_centers():
    xs = np.linspace(GRID_X[0], GRID_X[1], GRID_N + 1)
    zs = np.linspace(GRID_Z[0], GRID_Z[1], GRID_N + 1)
    return (xs[:-1] + xs[1:]) / 2, (zs[:-1] + zs[1:]) / 2


def grid_heatmap(g: pd.DataFrame, metric: str, title: str, min_n: int = 5, height=380, lisa_types=None):
    """격자 집계(gx, gz)를 히트맵으로. metric: density/whiff/xwoba/swing/cs."""
    xc, zc = grid_centers()
    Z = np.full((GRID_N, GRID_N), np.nan)
    N = np.zeros((GRID_N, GRID_N))
    gg = g.groupby(["gx", "gz"]).sum(numeric_only=True).reset_index()
    tot = gg["n"].sum()
    for _, r in gg.iterrows():
        i, j = int(r["gz"]), int(r["gx"])
        N[i, j] = r["n"]
        if metric == "density":
            Z[i, j] = r["n"] / tot * 100
        elif metric == "whiff" and r["swing"] >= min_n:
            Z[i, j] = r["whiff"] / r["swing"] * 100
        elif metric == "xwoba" and r["bip"] >= min_n:
            Z[i, j] = r["xwoba_bip"] / r["bip"]
        elif metric == "swing" and r["n"] >= min_n:
            Z[i, j] = r["swing"] / r["n"] * 100
        elif metric == "cs" and r["n"] >= min_n:
            Z[i, j] = r["called_strike"] / r["n"] * 100
    unit = {"density": "투구 비율 %", "whiff": "헛스윙률 %", "xwoba": "타구 xwOBA", "swing": "스윙률 %", "cs": "콜 스트라이크 %"}[metric]
    scale = SEQ if metric in ("density", "swing", "cs") else DIV
    zmid = None
    if metric == "xwoba":
        zmid = 0.37
    fig = go.Figure(go.Heatmap(x=xc, y=zc, z=Z, colorscale=scale, zmid=zmid, customdata=N, xgap=2, ygap=2,
                               colorbar=dict(title=unit, thickness=10),
                               hovertemplate="x %{x:.2f}ft, 높이 %{y:.2f}ft<br>" + unit + " %{z:.2f}<br>투구 %{customdata:.0f}<extra></extra>"))
    zone_shapes(fig)
    if lisa_types is not None:
        hh = lisa_types[lisa_types["유형"].isin(["HH", "LL"])]
        if len(hh):
            fig.add_trace(go.Scatter(x=xc[hh["gx"].astype(int)], y=zc[hh["gz"].astype(int)], mode="markers+text",
                                     text=hh["유형"], textfont=dict(color=INK, size=10),
                                     marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=INK, width=1.5)),
                                     hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title, height=height, xaxis=dict(range=[GRID_X[0], GRID_X[1]], title="포수 시점 좌우(ft)", showgrid=False),
                      yaxis=dict(range=[0, GRID_Z[1]], title="높이(ft)", showgrid=False, scaleanchor="x"), margin=dict(l=50, r=10))
    return fig


def note(text: str):
    st.markdown(f"<div class='small-note'>{text}</div>", unsafe_allow_html=True)
