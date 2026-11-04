import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import (PCFG, profile_block, CAT, DIV, GROUP_COLOR, INK, MUTED, PITCH_COLOR, SURFACE, fmt, grid_heatmap, ko, load, meta, names,
                 note, percentile_bars, pick_player, require_data, setup)
from mlbproj.analysis.spatial import grid_metric, lisa, rook_weights

@st.cache_data(show_spinner=False)
def league_xwoba_by_pitch(season):
    vp = load("batter_vs_pitch")
    d = vp[(vp["season"] == season) & vp["xwOBA"].notna()]
    w = d["PA"].clip(lower=0)
    return (d["xwOBA"] * w).groupby(d["key"]).sum() / w.groupby(d["key"]).sum()


@st.cache_data(show_spinner=False)
def lisa_cells(bid, season, grp, mk):
    bg = load("batter_grid")
    g = bg[(bg["batter"] == bid) & (bg["season"] == season)]
    gg = g if grp == "전체" else g[g["pitch_group"] == grp]
    mm = grid_metric(gg, mk, min_n=6)
    if len(mm) < 12:
        return None
    W = rook_weights(list(zip(mm["gx"], mm["gz"])))
    return pd.concat([mm[["gx", "gz"]].reset_index(drop=True), lisa(mm["value"].values, W, perms=299)], axis=1)


setup("🏏 타자 리포트")
require_data()
bid, season, row = pick_player("batter", "b")

st.subheader(f"{row['name']} · {season}")
tags = [{"L": "좌타", "R": "우타", "S": "스위치"}.get(row.get("stand"), "")]
cl = load("batter_clusters")
c_row = cl[(cl["batter"] == bid) & (cl["season"] == season)]
cm = meta("clusters_meta").get("batter", {})
if len(c_row):
    t = c_row.iloc[0]["유형"]
    tags.append(f"유형 {t}: {cm.get('desc', {}).get(t, '')}")
st.markdown(" ".join(f"<span class='badge'>{x}</span>" for x in tags if x), unsafe_allow_html=True)

cards = [("타율", "AVG", "rate"), ("출루율", "OBP", "rate"), ("장타율", "SLG", "rate"), ("OPS", "OPS", "rate"),
         ("홈런", "홈런", "int"), ("도루", "도루", "int"), ("wRC+", "wRC+", "int"), ("fWAR", "fWAR", "num"),
         ("wOBA", "wOBA", "rate"), ("xwOBA", "xwOBA", "rate")]
cols = st.columns(len(cards))
for c, (lab, key, k) in zip(cols, cards):
    v = row.get(key)
    c.metric(lab, "–" if v is None or pd.isna(v) else (f"{int(round(v))}" if k == "int" else fmt(v, k, 1)))
notes = []
if row.get("wRC+ 출처") == "자체 계산":
    notes.append("wRC+는 FanGraphs 값을 받지 못해 wOBA로 직접 계산한 추정치입니다(구장 보정 없음, 100 = 리그 평균).")
if pd.isna(row.get("fWAR")):
    notes.append("fWAR은 FanGraphs에서만 제공되어, 자동 수집이 막히면 비어 있습니다.")
if notes:
    note(" ".join(notes))

st.divider()
view = st.segmented_control("보기", ["요약", "구종별", "존 약점", "구속·카운트·좌우", "스프레이·유사 타자"], default="요약", key="view_batter")
view = view or "요약"

if view == "요약":
    # ------------------------------------------------ 요약
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(percentile_bars(row, [
            ("xwOBA", "xwOBA", "rate"), ("xBA", "xBA", "rate"), ("xSLG", "xSLG", "rate"), ("평균 타구속도", "평균 타구속도", "num"),
            ("EV90", "EV90", "num"), ("HardHit%", "HardHit%", "pct"), ("Barrel%", "Barrel%", "pct"), ("배트 스피드", "배트 스피드", "num"),
            ("K%", "K%", "pct"), ("BB%", "BB%", "pct"), ("Chase%", "Chase%", "pct"), ("Whiff%", "Whiff%", "pct")]),
            width="stretch", config=PCFG)
    with right:
        st.markdown("##### 타구와 스윙")
        tab = pd.DataFrame([{
            "평균 타구속도": row.get("평균 타구속도"), "최고 타구속도": row.get("최고 타구속도"), "평균 발사각": row.get("평균 발사각"),
            "스윗스팟%": row.get("SweetSpot%", np.nan) * 100, "배트 스피드": row.get("배트 스피드"), "스윙 길이": row.get("스윙 길이")}])
        st.dataframe(tab.round(1), hide_index=True)
        dirs = pd.DataFrame({"방향": ["당겨치기", "가운데", "밀어치기"],
                             "비율": [row.get("당겨치기%"), row.get("가운데%"), row.get("밀어치기%")]})
        fig = go.Figure(go.Bar(x=dirs["방향"], y=dirs["비율"] * 100, marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                               text=[f"{v * 100:.0f}%" if pd.notna(v) else "" for v in dirs["비율"]], textposition="outside"))
        fig.update_layout(title="타구 방향", height=250, yaxis=dict(title="%", range=[0, 70]))
        st.plotly_chart(fig, width="stretch", config=PCFG)
        lb = load("luck_batter")
        l = lb[(lb["batter"] == bid) & (lb["season"] == season)]
        if len(l):
            l = l.iloc[0]
            sig = "유의 (FDR 5%)" if l["유의(FDR 5%)"] else "유의하지 않음"
            st.markdown(f"**기대 성적** wOBA {l['wOBA']:.3f} − xwOBA {l['xwOBA']:.3f} = **{l['차이']:+.3f}** ({sig}). "
                        "−면 맞힌 공의 질에 비해 결과가 덜 나온 것(반등 후보).")

    st.markdown("##### 프로필 레이더와 종합 지수")
    profile_block("batter", bid, season, row,
                  [("xwOBA", "xwOBA"), ("Barrel%", "배럴"), ("HardHit%", "강한 타구"), ("평균 타구속도", "타구속도"),
                   ("BB%", "볼넷"), ("K%", "삼진 억제"), ("Chase%", "선구안"), ("배트 스피드", "배트 스피드")],
                  c_row.iloc[0]["유형"] if len(c_row) else None)

if view == "구종별":
    # ------------------------------------------------ 구종별
    st.subheader("구종별 상대 성적")
    vp = load("batter_vs_pitch")
    v = vp[(vp["batter"] == bid) & (vp["season"] == season) & (vp["pitches"] >= 15)].sort_values("pitches", ascending=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[ko(p) for p in v["key"]], y=v["xwOBA"], marker=dict(color=[PITCH_COLOR.get(p, MUTED) for p in v["key"]], line=dict(color=SURFACE, width=2)),
                             text=[f"{x:.3f}" if pd.notna(x) else "" for x in v["xwOBA"]], textposition="outside",
                             customdata=v["PA"], hovertemplate="%{x}: xwOBA %{y:.3f}<br>%{customdata}타석<extra></extra>"))
        lg = league_xwoba_by_pitch(season)
        fig.add_trace(go.Scatter(x=[ko(p) for p in v["key"]], y=[lg.get(p) for p in v["key"]], mode="markers", name="리그 평균",
                                 marker=dict(symbol="line-ew", size=26, line=dict(color=INK, width=2))))
        fig.update_layout(title="구종별 xwOBA (선 = 리그 평균)", height=330, yaxis_title="xwOBA", showlegend=False)
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        tab = v[["key", "pitches", "PA", "타율", "장타율", "xwOBA", "Swing%", "Whiff%", "Chase%", "HardHit%", "RV/100"]].copy()
        tab["key"] = tab["key"].map(ko)
        for c in ("Swing%", "Whiff%", "Chase%", "HardHit%"):
            tab[c] = tab[c] * 100
        st.dataframe(tab.rename(columns={"key": "구종", "pitches": "본 공"}), hide_index=True, column_config={
            **{c: st.column_config.NumberColumn(format="%.3f") for c in ("타율", "장타율", "xwOBA")},
            **{c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Swing%", "Whiff%", "Chase%", "HardHit%")},
            "RV/100": st.column_config.NumberColumn(format="%.2f")})

if view == "존 약점":
    # ------------------------------------------------ 존
    st.subheader("존별 약점 지도")
    c1, c2 = st.columns([1, 1])
    bg = load("batter_grid")
    g = bg[(bg["batter"] == bid) & (bg["season"] == season)]
    with c1:
        grp = st.radio("구종 그룹", ["전체", "패스트볼", "브레이킹", "오프스피드"], horizontal=True)
        metric = st.radio("지표", ["헛스윙률", "타구 xwOBA", "스윙률"], horizontal=True)
        show_lisa = st.checkbox("LISA 핫스팟 표시 (HH = 높은 칸끼리 뭉침, LL = 낮은 칸끼리 뭉침)", value=True)
        gg = g if grp == "전체" else g[g["pitch_group"] == grp]
        mk = {"헛스윙률": "whiff", "타구 xwOBA": "xwoba", "스윙률": "swing"}[metric]
        lt = lisa_cells(bid, season, grp, mk) if show_lisa and len(gg) else None
        st.plotly_chart(grid_heatmap(gg, mk, f"{grp} · {metric}", lisa_types=lt), width="stretch", config=PCFG)
    with c2:
        vz = load("batter_vs_zone")
        z = vz[(vz["batter"] == bid) & (vz["season"] == season)].copy()
        z["key"] = pd.to_numeric(z["key"], errors="coerce")
        zm = st.radio("9분할 존 지표", ["xwOBA", "Whiff%", "Swing%", "타율"], horizontal=True)
        Z = np.full((3, 3), np.nan)
        N = np.zeros((3, 3))
        for _, r in z[z["key"].between(1, 9)].iterrows():
            k = int(r["key"]) - 1
            Z[k // 3, k % 3] = r[zm] * (100 if zm in ("Whiff%", "Swing%") else 1)
            N[k // 3, k % 3] = r["pitches"]
        fig = go.Figure(go.Heatmap(z=Z, x=["왼쪽", "가운데", "오른쪽"], y=["높음", "가운데", "낮음"], colorscale=DIV,
                                   text=np.where(np.isnan(Z), "", np.vectorize(lambda x: f"{x:.3f}" if zm in ("xwOBA", "타율") else f"{x:.0f}%")(np.nan_to_num(Z))),
                                   texttemplate="%{text}", textfont=dict(size=15), customdata=N, xgap=3, ygap=3,
                                   hovertemplate="%{y}·%{x}<br>%{z:.3f}<br>%{customdata}구<extra></extra>", colorbar=dict(thickness=10)))
        fig.update_layout(title=f"Statcast 9분할 존 · {zm} (포수 시점: 우타자는 왼쪽이 몸쪽)", height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", config=PCFG)
        mo = load("batter_moran")
        m = mo[(mo["batter"] == bid) & (mo["season"] == season)]
        if len(m):
            m = m.iloc[0]
            note(f"헛스윙률 지도의 Global Moran's I = {m['MoranI_whiff']:.3f} (순열검정 p = {m['p']:.3f}). "
                 "양수이고 유의하면 약점이 특정 구역에 뭉쳐 있다는 뜻입니다.")

if view == "구속·카운트·좌우":
    # ------------------------------------------------ 구속대·카운트·좌우
    c1, c2, c3 = st.columns(3)
    with c1:
        vv = load("batter_vs_velo")
        d = vv[(vv["batter"] == bid) & (vv["season"] == season)].set_index("key").reindex(["~92", "92~95", "95~97", "97+"])
        fig = go.Figure(go.Bar(x=d.index, y=d["xwOBA"], marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                               text=[f"{x:.3f}" if pd.notna(x) else "" for x in d["xwOBA"]], textposition="outside",
                               customdata=d["pitches"], hovertemplate="%{x} mph: %{y:.3f}<br>%{customdata}구<extra></extra>"))
        fig.update_layout(title="패스트볼 구속대별 xwOBA", height=300, xaxis_title="mph", yaxis_title="xwOBA")
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        vc = load("batter_vs_count")
        order = ["초구", "타자 유리", "동등/투수 유리", "2스트라이크", "풀카운트"]
        d = vc[(vc["batter"] == bid) & (vc["season"] == season)].set_index("key").reindex(order)
        fig = go.Figure(go.Bar(x=d.index, y=d["Swing%"] * 100, marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                               text=[f"{x * 100:.0f}%" if pd.notna(x) else "" for x in d["Swing%"]], textposition="outside"))
        fig.update_layout(title="볼카운트별 스윙률", height=300, yaxis=dict(title="%", range=[0, 100]))
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c3:
        vh = load("batter_vs_hand")
        d = vh[(vh["batter"] == bid) & (vh["season"] == season)].set_index("key")
        if len(d):
            t = d[["PA", "타율", "장타율", "xwOBA", "K%", "BB%"]].rename(index={"L": "좌투 상대", "R": "우투 상대"})
            for c in ("K%", "BB%"):
                t[c] = (t[c] * 100).round(1)
            st.markdown("##### 좌·우투수 상대")
            st.dataframe(t.round(3))

if view == "스프레이·유사 타자":
    # ------------------------------------------------ 스프레이
    st.subheader("스프레이 차트")
    sp = load("batter_spray")
    s = sp[(sp["batter"] == bid) & (sp["season"] == season)].copy()
    if len(s):
        s["x_ft"], s["y_ft"] = s["x"].astype(float) * 2.5, s["y"].astype(float) * 2.5
        res_map = {"single": ("1루타", CAT[0]), "double": ("2루타", CAT[2]), "triple": ("3루타", CAT[3]), "home_run": ("홈런", CAT[7]),
                   "out": ("아웃", "#c3c2b7")}
        fig = go.Figure()
        ang = np.radians(np.linspace(-45, 45, 60))
        fig.add_trace(go.Scatter(x=np.r_[0, 330 * np.sin(ang[:1]), 400 * np.sin(ang), 330 * np.sin(ang[-1:]), 0],
                                 y=np.r_[0, 330 * np.cos(ang[:1]), 400 * np.cos(ang), 330 * np.cos(ang[-1:]), 0],
                                 mode="lines", line=dict(color="#c3c2b7", width=1), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 63.6, 0, -63.6, 0], y=[0, 63.6, 127.3, 63.6, 0], mode="lines", line=dict(color="#c3c2b7", width=1),
                                 hoverinfo="skip", showlegend=False))
        for r in ["out", "single", "double", "triple", "home_run"]:
            d = s[s["result"] == r]
            if d.empty:
                continue
            nm, col = res_map[r]
            fig.add_trace(go.Scattergl(x=d["x_ft"], y=d["y_ft"], mode="markers", name=f"{nm} ({len(d)})",
                                     marker=dict(size=8 if r != "out" else 6, color=col, line=dict(color=SURFACE, width=1),
                                                 opacity=.9 if r != "out" else .6),
                                     customdata=d["ev"].astype(float), hovertemplate=nm + "<br>타구속도 %{customdata:.1f} mph<extra></extra>"))
        fig.update_layout(height=520, xaxis=dict(visible=False, range=[-320, 320]), yaxis=dict(visible=False, range=[-10, 440], scaleanchor="x"),
                          title="타구 분포 (포수 뒤에서 본 시점)", legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch", config=PCFG)

    sim = load("batter_similar")
    s = sim[(sim["batter"] == bid) & (sim["season"] == season)].copy()
    if len(s):
        s["선수"] = s["유사선수"].map(names("batter"))
        st.markdown("##### 비슷한 타자 Top 5")
        st.dataframe(s[["순위", "선수", "유사도"]], hide_index=True,
                     column_config={"유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
