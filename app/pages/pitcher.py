import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import (PCFG, INK2, MUTED, PITCH_COLOR, SEQ, SURFACE, fmt, grid_heatmap, ko, load, meta, names, note,
                 percentile_bars, pick_player, require_data, setup)

setup("🎯 투수 리포트")
require_data()
pid, season, row = pick_player("pitcher", "p")

# ------------------------------------------------ 성적 카드
st.subheader(f"{row['name']} · {season}")
tags = [f"{'좌투' if row.get('p_throws') == 'L' else '우투'}", f"{int(row['구종 수']) if pd.notna(row.get('구종 수')) else '–'}개 구종"]
cl = load("pitcher_clusters")
c_row = cl[(cl["pitcher"] == pid) & (cl["season"] == season)]
cm = meta("clusters_meta").get("pitcher", {})
if len(c_row):
    t = c_row.iloc[0]["유형"]
    tags.append(f"유형 {t}: {cm.get('desc', {}).get(t, '')}")
st.markdown(" ".join(f"<span class='badge'>{x}</span>" for x in tags), unsafe_allow_html=True)

cards = [("ERA", "ERA", "num", 2), ("FIP", "FIP", "num", 2), ("이닝", "IP", "num", 1), ("WHIP", "WHIP", "num", 2),
         ("fWAR", "fWAR", "num", 1), ("K%", "K%", "pct", 1), ("BB%", "BB%", "pct", 1), ("xwOBA", "xwOBA", "rate", 0),
         ("Whiff%", "Whiff%", "pct", 1), ("FB 구속", "패스트볼 평균구속", "num", 1)]
cols = st.columns(len(cards))
for c, (lab, key, k, d) in zip(cols, cards):
    c.metric(lab, fmt(row.get(key), k, d))
if "fWAR" not in row or pd.isna(row.get("fWAR")):
    note("fWAR은 FanGraphs CSV를 data/external 에 넣으면 표시됩니다 (README 참고).")
ar = load("pitcher_arsenal")
a = ar[(ar["pitcher"] == pid) & (ar["season"] == season)].sort_values("pitches", ascending=False)
stuff = load("stuff_scores")
a = a.merge(stuff[["pitcher", "season", "pitch_type", "구위 지수"]], on=["pitcher", "season", "pitch_type"], how="left")
st.divider()
view = st.segmented_control("보기", ["요약", "구종 구성", "구종별 성적", "투구 위치", "볼배합", "구속 변화"], default="요약", key="view_pitcher")
view = view or "요약"

if view == "요약":
    # ------------------------------------------------ 요약
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(percentile_bars(row, [
            ("RV/100", "구종가치/100구", "num"), ("xwOBA", "xwOBA", "rate"), ("K%", "K%", "pct"), ("BB%", "BB%", "pct"),
            ("Whiff%", "Whiff%", "pct"), ("Chase%", "Chase%", "pct"), ("CSW%", "CSW%", "pct"),
            ("HardHit%", "HardHit%", "pct"), ("Barrel%", "Barrel%", "pct"), ("GB%", "GB%", "pct"),
            ("패스트볼 평균구속", "패스트볼 구속", "num")]), width="stretch", config=PCFG)
    with right:
        era_ = load("era_residuals")
        lp = load("luck_pitcher")
        e = era_[(era_["pitcher"] == pid) & (era_["season"] == season)]
        l = lp[(lp["pitcher"] == pid) & (lp["season"] == season)]
        st.markdown("##### 기대 성적 갭")
        if len(e):
            e = e.iloc[0]
            st.markdown(f"- 실제 ERA **{e['ERA']:.2f}** vs 기대 ERA(삼진·볼넷·땅볼·강한 타구로 추정) **{e['기대 ERA']:.2f}** → 차이 **{e['ERA 잔차']:+.2f}**")
        if len(l):
            l = l.iloc[0]
            sig = "통계적으로 유의 (FDR 5%)" if l["유의(FDR 5%)"] else "유의하지 않음"
            st.markdown(f"- 허용 wOBA **{l['wOBA']:.3f}** vs xwOBA **{l['xwOBA']:.3f}** → 차이 **{l['차이']:+.3f}** ({sig})")
            st.markdown("- 차이가 양수면 타구 질보다 결과가 나빴던 것(불운 → 반등 후보), 음수면 기대보다 결과가 좋았던 것(하락 주의)")
        st.markdown("##### 기본 기록")
        show = {k: row.get(k) for k in ("경기", "선발", "승", "패", "세이브", "IP", "PA", "pitches", "HR") if k in row}
        st.dataframe(pd.DataFrame([show]).astype(object).where(pd.notna(pd.DataFrame([show])), "–").rename(columns={"IP": "이닝", "pitches": "투구 수", "HR": "피홈런"}), hide_index=True)
        sim = load("pitcher_similar")
        s = sim[(sim["pitcher"] == pid) & (sim["season"] == season)].copy()
        if len(s):
            nm = names("pitcher")
            s["선수"] = s["유사선수"].map(nm)
            st.markdown("##### 비슷한 투수 Top 5 (코사인 유사도)")
            st.dataframe(s[["순위", "선수", "유사도"]], hide_index=True, column_config={
                "유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

if view == "구종 구성":
    # ------------------------------------------------ 구종 구성
    st.subheader("구종 구성")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        for side, col in (("좌타자 상대", "좌타 구사율"), ("우타자 상대", "우타 구사율"), ("전체", "구사율")):
            for _, r in a.iterrows():
                fig.add_trace(go.Bar(y=[side], x=[r[col] * 100], name=ko(r["pitch_type"]), orientation="h",
                                     marker=dict(color=PITCH_COLOR.get(r["pitch_type"], MUTED), line=dict(color=SURFACE, width=2)),
                                     legendgroup=r["pitch_type"], showlegend=side == "전체",
                                     text=f"{ko(r['pitch_type'])} {r[col] * 100:.0f}%" if r[col] >= .08 else "",
                                     textposition="inside", insidetextanchor="middle", textfont=dict(color="#ffffff"),
                                     hovertemplate=f"{ko(r['pitch_type'])}: %{{x:.1f}}%<extra>{side}</extra>"))
        fig.update_layout(barmode="stack", title="구사율 (좌·우타자별)", height=280, xaxis=dict(range=[0, 100], title="%"))
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        hist = ar[ar["pitcher"] == pid].pivot_table(index="season", columns="pitch_type", values="구사율").fillna(0)
        fig = go.Figure()
        for pt in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[pt] * 100, name=ko(pt), mode="lines+markers",
                                     line=dict(color=PITCH_COLOR.get(pt, MUTED), width=2), marker=dict(size=8)))
        fig.update_layout(title="시즌별 구사율 변화", height=280, yaxis_title="%", xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch", config=PCFG)

    # 무브먼트
    c1, c2 = st.columns([1, 1])
    with c1:
        lg = ar[(ar["season"] == season) & (ar["pitches"] >= 50)]
        fig = go.Figure()
        for pt in a["pitch_type"]:
            L = lg[lg["pitch_type"] == pt]
            fig.add_trace(go.Scattergl(x=L["수평무브"], y=L["수직무브"], mode="markers", marker=dict(size=5, color=PITCH_COLOR.get(pt, MUTED), opacity=.12),
                                     hoverinfo="skip", showlegend=False))
        for _, r in a.iterrows():
            fig.add_trace(go.Scatter(x=[r["수평무브"]], y=[r["수직무브"]], mode="markers+text", name=ko(r["pitch_type"]),
                                     text=[ko(r["pitch_type"])], textposition="top center", textfont=dict(color=INK2),
                                     marker=dict(size=16, color=PITCH_COLOR.get(r["pitch_type"], MUTED), line=dict(color=SURFACE, width=2)),
                                     hovertemplate=f"{ko(r['pitch_type'])}<br>수평 %{{x:.1f}}in · 수직 %{{y:.1f}}in<br>구속 {r['구속']:.1f} · 회전 {r['회전수']:.0f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1))
        fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
        fig.update_layout(title="무브먼트 (진한 점 = 이 투수, 옅은 점 = 같은 구종의 리그 투수들)", height=420, showlegend=False,
                          xaxis_title="수평 무브먼트(인치, + = 팔 쪽)", yaxis_title="수직 무브먼트(인치, 중력 제외)")
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        phys = a[["pitch_type", "구사율", "구속", "최고구속", "회전수", "수직무브", "수평무브", "익스텐션", "릴리스높이", "패스트볼 대비 구속차", "수직무브_sd", "구위 지수"]].copy()
        phys["pitch_type"] = phys["pitch_type"].map(ko)
        phys["구사율"] *= 100
        st.markdown("##### 구종별 물리 특성")
        phys = phys.rename(columns={"pitch_type": "구종", "구사율": "구사율%", "수직무브_sd": "수직무브 일관성(SD)"})
        st.dataframe(phys, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.1f") for c in phys.columns if c != "구종"})
        note("구위 지수: 구종의 물리 특성만으로 예측한 헛스윙률을 같은 구종 그룹 리그 평균 대비로 나타낸 값(100 = 평균). 수직무브 SD가 작을수록 일정하게 던짐.")

if view == "구종별 성적":
    # ------------------------------------------------ 구종별 성적
    st.subheader("구종별 성적")
    perf = a[["pitch_type", "pitches", "PA", "타율", "장타율", "wOBA", "xBA", "xwOBA", "Whiff%", "CSW%", "Chase%", "Zone%",
              "HardHit%", "Barrel%", "평균 타구속도", "평균 발사각", "GB%", "RV/100", "RV"]].copy()
    perf["pitch_type"] = perf["pitch_type"].map(ko)
    pc = {c: st.column_config.NumberColumn(format="%.3f") for c in ("타율", "장타율", "wOBA", "xBA", "xwOBA")}
    pc.update({c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Whiff%", "CSW%", "Chase%", "Zone%", "HardHit%", "Barrel%", "GB%")})
    for c in ("Whiff%", "CSW%", "Chase%", "Zone%", "HardHit%", "Barrel%", "GB%"):
        perf[c] = perf[c] * 100
    pc.update({c: st.column_config.NumberColumn(format="%.1f") for c in ("평균 타구속도", "평균 발사각", "RV/100", "RV")})
    st.dataframe(perf.rename(columns={"pitch_type": "구종", "pitches": "투구"}), hide_index=True, column_config=pc, width="stretch")
    note("RV/100: 100구당 실점 억제량(Statcast delta_run_exp 기반, +가 투수에게 좋음). 결과(타율·wOBA)와 기대값(xBA·xwOBA) 차이가 크면 결과가 과정과 어긋난 것 영향.")

if view == "투구 위치":
    # ------------------------------------------------ 위치
    st.subheader("구종별 투구 위치")
    grid = load("pitcher_grid")
    g = grid[(grid["pitcher"] == pid) & (grid["season"] == season)]
    if g.empty:
        st.info("이 시즌은 위치 격자를 만들 표본이 부족합니다.")
    else:
        c1, c2, c3 = st.columns(3)
        pts = [p for p in a["pitch_type"] if p in set(g["pitch_type"])]
        pt = c1.selectbox("구종", pts, format_func=ko)
        side = c2.radio("상대 타자", ["전체", "좌타", "우타"], horizontal=True)
        metric = c3.radio("표시", ["투구 분포", "헛스윙률", "타구 xwOBA"], horizontal=True)
        gg = g[g["pitch_type"] == pt]
        if side != "전체":
            gg = gg[gg["stand"] == ("L" if side == "좌타" else "R")]
        mkey = {"투구 분포": "density", "헛스윙률": "whiff", "타구 xwOBA": "xwoba"}[metric]
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(grid_heatmap(gg, mkey, f"{ko(pt)} · {side} · {metric} (포수 시점)"), width="stretch", config=PCFG)
        with c2:
            sp = load("pitcher_splits")
            s = sp[(sp["pitcher"] == pid) & (sp["season"] == season)].set_index("stand")
            if len(s):
                st.markdown("##### 좌·우타자 상대 성적")
                tab = s[["PA", "K%", "BB%", "xwOBA", "Whiff%", "HardHit%"]].rename(index={"L": "좌타자", "R": "우타자"})
                for c in ("K%", "BB%", "Whiff%", "HardHit%"):
                    tab[c] = (tab[c] * 100).round(1)
                tab["xwOBA"] = tab["xwOBA"].round(3)
                st.dataframe(tab)
            note("격자 한 칸 = 0.4ft(약 12cm). 굵은 사각형 = 평균 스트라이크존, 점선 = 9분할.")

if view == "볼배합":
    # ------------------------------------------------ 볼배합
    st.subheader("볼배합")
    c1, c2 = st.columns([1, 1])
    with c1:
        cu = load("pitcher_count_usage")
        u = cu[(cu["pitcher"] == pid) & (cu["season"] == season)]
        order = ["초구", "타자 유리", "동등/투수 유리", "2스트라이크", "풀카운트"]
        fig = go.Figure()
        for pt in a["pitch_type"]:
            d = u[u["pitch_type"] == pt].set_index("count_state").reindex(order)
            fig.add_trace(go.Bar(y=order, x=d["share"] * 100, name=ko(pt), orientation="h",
                                 marker=dict(color=PITCH_COLOR.get(pt, MUTED), line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{ko(pt)} %{{x:.1f}}%<extra>%{{y}}</extra>"))
        fig.update_layout(barmode="stack", title="볼카운트 상황별 구사율", height=340, xaxis=dict(range=[0, 100], title="%"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        tr = load("pitcher_transitions")
        t = tr[(tr["pitcher"] == pid) & (tr["season"] == season)]
        if len(t):
            M = t.pivot_table(index="from", columns="to", values="n", fill_value=0)
            types = [p for p in a["pitch_type"] if p in M.index and p in M.columns]
            M = M.loc[types, types]
            P = M.div(M.sum(axis=1), axis=0)
            ent = -(P * np.log2(P.replace(0, np.nan))).sum(axis=1).mean()
            fig = go.Figure(go.Heatmap(z=P.values * 100, x=[ko(x) for x in types], y=[ko(x) for x in types], colorscale=SEQ,
                                       text=np.round(P.values * 100).astype(int), texttemplate="%{text}%", xgap=2, ygap=2,
                                       colorbar=dict(title="%", thickness=10),
                                       hovertemplate="%{y} 다음 %{x}: %{z:.1f}%<extra></extra>"))
            fig.update_layout(title=f"구종 전이 확률 (마르코프) · 평균 엔트로피 {ent:.2f} bit", height=340,
                              xaxis_title="다음 공", yaxis=dict(title="이전 공", autorange="reversed"))
            st.plotly_chart(fig, width="stretch", config=PCFG)
            note("엔트로피가 높을수록 다음 공을 예측하기 어려운 투수. 같은 타석 안의 연속 투구만 사용.")

if view == "구속 변화":
    # ------------------------------------------------ 시즌 내 구속
    st.subheader("시즌 중 구속 변화와 변화점")
    gv = load("pitcher_monthly")
    m = gv[(gv["pitcher"] == pid) & (gv["season"] == season)]
    cp = load("velo_changepoints")
    c = cp[(cp["pitcher"] == pid) & (cp["season"] == season)]
    if len(m):
        fig = go.Figure(go.Scatter(x=m["month"], y=m["fb_velo"], mode="lines+markers", line=dict(color=PITCH_COLOR["FF"], width=2),
                                   marker=dict(size=8), name="월별 패스트볼 평균", hovertemplate="%{x}월 %{y:.1f} mph<extra></extra>"))
        fig.update_layout(title="월별 패스트볼 평균 구속", height=300, xaxis=dict(title="월", dtick=1), yaxis_title="mph", showlegend=False)
        st.plotly_chart(fig, width="stretch", config=PCFG)
    if len(c):
        c = c.iloc[0]
        msg = f"가장 뚜렷한 변화점: **{pd.to_datetime(c['변화 시점']):%m월 %d일}** 전후 {c['이전 평균']:.1f} → {c['이후 평균']:.1f} mph ({c['변화량']:+.1f}, t = {c['t']:.1f})"
        (st.error if c["구속 급락 경보"] else st.info)(msg + ("  ⚠ 구속 급락 경보 (−1.5mph 이상, t ≤ −4)" if c["구속 급락 경보"] else ""))
