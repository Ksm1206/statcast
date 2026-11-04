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
# 툴바를 숨겨 차트마다 붙는 DOM을 줄임 (스크롤 버벅임 완화)
PCFG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}

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


ALL_TEAMS = "전체 구단"  # 내부용 (선택지에는 표시하지 않음)


def team_list(sub: pd.DataFrame) -> list[str]:
    """시즌 중 이적한 선수는 '팀A/팀B'로 저장되어 있어 각 팀에 모두 포함."""
    teams = set()
    for t in sub.get("팀", pd.Series(dtype=object)).dropna():
        teams |= {x.strip() for x in str(t).split("/") if x.strip()}
    return sorted(teams)


def filter_team(sub: pd.DataFrame, team: str) -> pd.DataFrame:
    if team == ALL_TEAMS:
        return sub
    return sub[sub["팀"].fillna("").astype(str).str.split("/").apply(lambda xs: team in [x.strip() for x in xs])]


def team_player_select(sub: pd.DataFrame, kind: str, key: str, label: str, col_team, col_player, stop: bool = True):
    """구단을 먼저 고르고, 그 구단 선수만 선택. 반환: player_id (구단 미선택이면 화면 중단)"""
    teams = team_list(sub)
    team = col_team.selectbox("구단", teams, index=None, placeholder="구단을 선택하세요", key=f"{key}_team")
    if team is None:
        col_player.selectbox(label, [], placeholder="먼저 구단을 선택하세요", disabled=True, key=f"{key}_pid_off")
        if not stop:
            return None
        st.info(f"구단을 선택하면 {label} 목록이 나옵니다.")
        st.stop()
    s2 = filter_team(sub, team)
    if s2.empty:
        col_player.warning("조건에 맞는 선수가 없습니다.")
        st.stop()
    lab = dict(zip(s2[kind], s2["name"]))
    return col_player.selectbox(label, s2[kind].tolist(), format_func=lambda i: lab.get(i, str(i)), key=f"{key}_pid")


def pick_player(kind: str, key: str, label: str | None = None, default_top: bool = True):
    """시즌 → 구단 → 선수 선택. 반환: (player_id, season, row)."""
    t = load(f"{kind}_season")
    seasons = sorted(t["season"].unique(), reverse=True)
    c1, c2, c3 = st.columns([1, 1.6, 2.4])
    season = c1.selectbox("시즌", seasons, key=f"{key}_season")
    sub = t[t["season"] == season].sort_values("pitches" if kind == "pitcher" else "PA", ascending=False)
    sub = sub[sub["qualified"]] if c1.checkbox("규정 표본만", True, key=f"{key}_q") else sub
    pid = team_player_select(sub, kind, key, label or ("투수" if kind == "pitcher" else "타자"), c2, c3)
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


# ---------------------------------------------------------------- 레이더 차트
def radar(series: list[tuple[str, list[float], str]], axes: list[str], title: str, height: int = 380):
    """series: (이름, 0~100 값 목록, 색). 2개 이상이면 범례 표시."""
    fig = go.Figure()
    for name, vals, col in series:
        v = [0 if (x is None or pd.isna(x)) else float(x) for x in vals]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], name=name, mode="lines+markers",
                                      line=dict(color=col, width=2), marker=dict(size=6, color=col),
                                      fill="toself", fillcolor=_alpha(col, 0.12),
                                      hovertemplate="%{theta}: %{r:.0f}<extra>" + name + "</extra>"))
    fig.update_layout(title=title, height=height, showlegend=len(series) > 1,
                      polar=dict(bgcolor=SURFACE, radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], gridcolor=GRID,
                                                                   tickfont=dict(color=MUTED, size=10), angle=90),
                                 angularaxis=dict(gridcolor=GRID, tickfont=dict(color=INK2, size=11))),
                      margin=dict(l=40, r=40, t=50, b=30))
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ---------------------------------------------------------------- 볼배합 네트워크
def network_figure(t: pd.DataFrame, types: list[str], nodes: pd.DataFrame, title: str, min_prob: float = 0.08):
    """t: from/to/n/swing/whiff 전이표, nodes: pitch_type/pagerank. 원형 배치 + 화살표 간선."""
    n = len(types)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = {p: (np.cos(a), np.sin(a)) for p, a in zip(types, ang)}
    out = t.groupby("from")["n"].sum()
    fig = go.Figure()
    ann = []
    for _, e in t.iterrows():
        a, b = e["from"], e["to"]
        if a not in pos or b not in pos or a == b:
            continue
        prob = e["n"] / out.get(a, 1)
        if prob < min_prob:
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy) or 1
        ox, oy = -dy / L * 0.06, dx / L * 0.06          # 양방향 간선이 겹치지 않게 옆으로 띄움
        sx, sy, ex, ey = x0 + dx * .16 + ox, y0 + dy * .16 + oy, x1 - dx * .16 + ox, y1 - dy * .16 + oy
        wr = e["whiff"] / e["swing"] if e["swing"] > 0 else np.nan
        ann.append(dict(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                        arrowhead=2, arrowsize=1, arrowwidth=1 + 7 * prob, arrowcolor=_alpha(PITCH_COLOR.get(a, MUTED), .75)))
        fig.add_trace(go.Scatter(x=[(sx + ex) / 2], y=[(sy + ey) / 2], mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                                 hovertemplate=f"{ko(a)} → {ko(b)}<br>전이 확률 {prob * 100:.0f}% ({int(e['n'])}회)"
                                               f"<br>도착 공 헛스윙률 {'' if pd.isna(wr) else f'{wr * 100:.0f}%'}<extra></extra>",
                                 showlegend=False))
    pr = nodes.set_index("pitch_type")["pagerank"]
    self_p = {p: (t[(t["from"] == p) & (t["to"] == p)]["n"].sum() / out.get(p, 1)) for p in types}
    fig.add_trace(go.Scatter(
        x=[pos[p][0] for p in types], y=[pos[p][1] for p in types], mode="markers+text",
        text=[f"{ko(p)}<br>{pr.get(p, 0):.2f}" for p in types], textposition="middle center",
        textfont=dict(color=[INK if PITCH_COLOR.get(p) in ("#eda100", "#a8a69e") else "#ffffff" for p in types], size=11),
        marker=dict(size=[34 + 90 * pr.get(p, 0) for p in types], color=[PITCH_COLOR.get(p, MUTED) for p in types],
                    line=dict(color=SURFACE, width=2)),
        customdata=[[pr.get(p, 0), self_p[p] * 100] for p in types],
        hovertemplate="%{text}<br>PageRank %{customdata[0]:.3f}<br>같은 구종 연속 %{customdata[1]:.0f}%<extra></extra>",
        showlegend=False))
    fig.update_layout(title=title, height=420, annotations=ann, xaxis=dict(visible=False, range=[-1.45, 1.45]),
                      yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def profile_block(kind: str, pid: int, season: int, row: pd.Series, axes: list[tuple[str, str]], type_label: str | None):
    """레이더(선수 vs 같은 유형 평균) + 종합 지수 카드."""
    cl = load("pitcher_types") if kind == "pitcher" else load(f"{kind}_clusters")
    t = load(f"{kind}_season")
    comp = load(f"composite_{kind}")
    cm = meta("composite_meta").get(kind, {})
    names_ = [a for _, a in axes]
    pv = [row.get(f"pct_{m}") for m, _ in axes]
    series = [(row["name"], pv, CAT[0])]
    if type_label is not None and len(cl):
        mem = cl[(cl["season"] == season) & (cl["유형"] == type_label)][[kind]]
        tt = t[(t["season"] == season)].merge(mem, on=kind)
        series.append(((f"{type_label} 평균" if kind == "pitcher" else f"유형 {type_label} 평균"), [tt[f"pct_{m}"].mean() if f"pct_{m}" in tt else np.nan for m, _ in axes], CAT[1]))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(radar(series, names_, "프로필 레이더 (리그 백분위, 바깥일수록 좋음)"), width="stretch", config=PCFG)
    with c2:
        r = comp[(comp[kind] == pid) & (comp["season"] == season)] if len(comp) else comp
        lab = "종합 투수 지수" if kind == "pitcher" else "종합 타자 지수"
        if len(r):
            r = r.iloc[0]
            st.metric(lab, f"{r['종합 지수']:.0f}점", f"시즌 상위 {100 - r['시즌 내 백분위']:.0f}%", delta_color="off")
            st.markdown(f"<span class='badge'>엔트로피 지수 {r['엔트로피 지수']:.0f}</span>"
                        f"<span class='badge'>SHAP 지수 {r['SHAP 지수']:.0f}</span>", unsafe_allow_html=True)
            note(f"최종 지수는 다음 시즌 예측력이 더 높았던 <b>{cm.get('final', '')}</b>를 사용합니다. 0~100점(그 시즌 규정 표본 최하 0, 최고 100). "
                 "자세한 가중치는 '종합 지수' 메뉴.")
        else:
            st.info(f"{lab}는 규정 표본 선수만 계산됩니다.")
