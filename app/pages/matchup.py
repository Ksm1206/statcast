import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import PCFG, ALL_TEAMS, CAT, DATA, filter_team, team_player_select, DIV, INK, MUTED, PITCH_COLOR, SURFACE, grid_heatmap, ko, load, meta, names, note, require_data, setup
from mlbproj.analysis.matchup import OUTCOMES, WOBA_W, predict_pairs

setup("⚔️ 맞대결 시뮬레이터")
require_data()
mm = meta("matchup_meta")
if not (DATA / "matchup_lgb.txt").exists():
    st.info("맞대결 모델 파일이 없습니다.")
    st.stop()


@st.cache_resource
def booster():
    return lgb.Booster(model_file=str(DATA / "matchup_lgb.txt"))


B = booster()
cols = mm["features"]
pf, bf, lg = load("matchup_features_pitcher"), load("matchup_features_batter"), load("matchup_league_rates")
ps, bs = load("pitcher_season"), load("batter_season")
pn, bn = names("pitcher"), names("batter")
LAB = {"K": "삼진", "BB": "볼넷·사구", "1B": "1루타", "XBH": "2·3루타", "HR": "홈런", "OUT": "범타 아웃"}

note("한 번도 안 만난 조합도 예측합니다. 선택한 시즌의 기록을 '직전 시즌 특성'으로 삼아 다음 맞대결을 예측하는 방식이며, "
     f"모델은 {mm.get('test_season')}시즌 전체로 검증했습니다 (검증표는 '검증 · 방법론' 메뉴).")

seasons = sorted(pf["season"].unique(), reverse=True)
season = st.selectbox("특성 기준 시즌", seasons)
pq = ps[(ps["season"] == season) & ps["qualified"]].sort_values("pitches", ascending=False)
bq = bs[(bs["season"] == season) & bs["qualified"]].sort_values("PA", ascending=False)
st.markdown("##### 투수")
c1, c2 = st.columns([2, 3])
pid = team_player_select(pq, "pitcher", "mp", "투수", c1, c2, stop=False)
st.markdown("##### 타자")
c1, c2 = st.columns([2, 3])
bid = team_player_select(bq, "batter", "mb", "타자", c1, c2, stop=False)
if pid is None or bid is None:
    st.info("투수와 타자의 구단을 각각 선택하면 선수 목록이 나옵니다.")
    st.stop()
p_throws = pq.set_index("pitcher").loc[pid, "p_throws"]
b_stand = bq.set_index("batter").loc[bid, "stand"]
stand = ("L" if p_throws == "R" else "R") if b_stand == "S" else b_stand

req = pd.DataFrame({"pitcher": [pid], "batter": [bid], "stand": [stand], "p_throws": [p_throws], "season": [season + 1]})
P, L, R = predict_pairs(req, pf, bf, lg, B, cols)
lgr = lg[lg["season"] == season][OUTCOMES].values[0]
mf_p, mf_b = load("matchup_mf_pitcher"), load("matchup_mf_batter")
mf = 0.0
if pid in set(mf_p["pitcher"]) and bid in set(mf_b["batter"]):
    u = mf_p.set_index("pitcher").loc[pid].values
    v = mf_b.set_index("batter").loc[bid].values
    mf = float(u @ v)
xw = float(P[0] @ WOBA_W)

st.subheader(f"{pn.get(pid)} ({'좌' if p_throws == 'L' else '우'}투)  vs  {bn.get(bid)} ({'좌' if stand == 'L' else '우'}타석)")
c = st.columns(4)
c[0].metric("예측 기대 wOBA (LightGBM)", f"{xw:.3f}", f"{xw - float(lgr @ WOBA_W):+.3f} vs 리그", delta_color="off")
c[1].metric("Log5 기대 wOBA", f"{float(L[0] @ WOBA_W):.3f}")
c[2].metric("행렬분해 궁합 보정", f"{mf:+.3f}")
c[3].metric("삼진 확률", f"{P[0][0] * 100:.1f}%", f"{(P[0][0] - lgr[0]) * 100:+.1f}%p vs 리그", delta_color="off")

fig = go.Figure()
for name, vals, col in (("LightGBM", P[0], CAT[0]), ("Log5", L[0], CAT[1]), ("리그 평균", lgr, "#c3c2b7")):
    fig.add_trace(go.Bar(x=[LAB[o] for o in OUTCOMES], y=vals * 100, name=name, marker=dict(color=col, line=dict(color=SURFACE, width=2)),
                         text=[f"{v * 100:.1f}" for v in vals], textposition="outside", hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>"))
fig.update_layout(barmode="group", title="타석 결과 확률", height=380, yaxis_title="%")
st.plotly_chart(fig, width="stretch", config=PCFG)

# ------------------------------------------------ 구종 조합 분석
st.subheader("구종 조합으로 본 맞대결")
ar = load("pitcher_arsenal")
a = ar[(ar["pitcher"] == pid) & (ar["season"] == season)].copy()
use_col = "좌타 구사율" if stand == "L" else "우타 구사율"
vp = load("batter_vs_pitch")
v = vp[(vp["batter"] == bid) & (vp["season"] == season)].set_index("key")
lgx = vp[vp["season"] == season].groupby("key").apply(lambda d: np.average(d["xwOBA"].fillna(.32), weights=d["PA"].clip(1)), include_groups=False)
rows = []
for _, r in a.iterrows():
    pt = r["pitch_type"]
    n_pa = v["PA"].get(pt, 0) if len(v) else 0
    bx = v["xwOBA"].get(pt, np.nan) if len(v) else np.nan
    base = lgx.get(pt, .32)
    shr = ((0 if pd.isna(bx) else bx) * n_pa + 40 * base) / (n_pa + 40)
    rows.append({"구종": ko(pt), "pt": pt, "투수 구사율": r[use_col], "투수 피xwOBA": r["xwOBA"], "타자 xwOBA(축소)": shr,
                 "타자 표본 PA": n_pa, "타자 Whiff%": v["Whiff%"].get(pt, np.nan) if len(v) else np.nan})
t = pd.DataFrame(rows)
if len(t):
    t["기여"] = t["투수 구사율"] * t["타자 xwOBA(축소)"]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t["구종"], y=t["타자 xwOBA(축소)"], name="타자 xwOBA (이 구종 상대)",
                             marker=dict(color=[PITCH_COLOR.get(p, MUTED) for p in t["pt"]], line=dict(color=SURFACE, width=2)),
                             text=[f"{x:.3f}" for x in t["타자 xwOBA(축소)"]], textposition="outside"))
        fig.add_trace(go.Scatter(x=t["구종"], y=t["투수 피xwOBA"], name="투수 피xwOBA (이 구종)", mode="markers",
                                 marker=dict(symbol="diamond", size=12, color=INK, line=dict(color=SURFACE, width=2))))
        fig.update_layout(title="구종별: 타자의 강점(막대) vs 투수의 강점(◆)", height=360, yaxis_title="xwOBA")
        st.plotly_chart(fig, width="stretch", config=PCFG)
    with c2:
        show = t[["구종", "투수 구사율", "타자 xwOBA(축소)", "타자 표본 PA", "타자 Whiff%", "기여"]].copy()
        show["투수 구사율"] *= 100
        show["타자 Whiff%"] *= 100
        st.dataframe(show, hide_index=True, column_config={
            "투수 구사율": st.column_config.NumberColumn(format="%.1f%%"), "타자 Whiff%": st.column_config.NumberColumn(format="%.1f%%"),
            "타자 xwOBA(축소)": st.column_config.NumberColumn(format="%.3f"), "기여": st.column_config.NumberColumn(format="%.3f")})
        st.metric("구종 조합 기대 xwOBA = Σ 구사율 × 타자 xwOBA", f"{t['기여'].sum() / t['투수 구사율'].sum():.3f}")
        note("타자 xwOBA는 표본이 작을수록 리그 평균 쪽으로 당겨 축소(경험적 베이즈, k=40 PA).")

c1, c2 = st.columns(2)
pg = load("pitcher_grid")
g1 = pg[(pg["pitcher"] == pid) & (pg["season"] == season) & (pg["stand"] == stand)]
bg = load("batter_grid")
g2 = bg[(bg["batter"] == bid) & (bg["season"] == season)]
if len(g1):
    c1.plotly_chart(grid_heatmap(g1, "density", f"투수가 {'좌' if stand == 'L' else '우'}타자에게 던지는 위치"), width="stretch", config=PCFG)
if len(g2):
    c2.plotly_chart(grid_heatmap(g2, "xwoba", "타자의 위치별 타구 xwOBA", min_n=3), width="stretch", config=PCFG)

# ------------------------------------------------ 실제 기록
h2h = load("h2h")
h = h2h[(h2h["pitcher"] == pid) & (h2h["batter"] == bid)]
st.subheader("실제 맞대결 기록 (정규시즌 누적)")
if h.empty:
    st.info("두 선수는 실제로 만난 기록이 없습니다. 위 예측은 각자의 특성만으로 계산한 값입니다.")
else:
    h = h.iloc[0]
    k = 50
    eb = (h["PA"] * (h["xwOBA"] if pd.notna(h["xwOBA"]) else xw) + k * xw) / (h["PA"] + k)
    c = st.columns(6)
    c[0].metric("타석", int(h["PA"]))
    c[1].metric("안타/타수", f"{int(h['H'])}/{int(h['AB'])}")
    c[2].metric("홈런", int(h["HR"]))
    c[3].metric("삼진 / 볼넷", f"{int(h['K'])} / {int(h['BB'])}")
    c[4].metric("실제 xwOBA", f"{h['xwOBA']:.3f}" if pd.notna(h["xwOBA"]) else "–")
    c[5].metric("축소 추정 xwOBA", f"{eb:.3f}")
    note(f"실제 기록 {int(h['PA'])}타석은 우연의 영향이 커서, 모델 예측과 섞어(k={k}타석) 보정한 값이 '축소 추정'입니다.")

# ------------------------------------------------ 최적 타순
st.divider()
st.subheader("이 투수 상대 최적 타자 9명")
from lib import team_list
teams = team_list(bq)
if teams:
    team = st.selectbox("상대 구단 (타선)", teams, key="lineup_team")
    roster = filter_team(bq, team)
    if len(roster):
        st_ = np.where(roster["stand"] == "S", "L" if p_throws == "R" else "R", roster["stand"])
        req = pd.DataFrame({"pitcher": pid, "batter": roster["batter"].values, "stand": st_, "p_throws": p_throws, "season": season + 1})
        P2, L2, _ = predict_pairs(req, pf, bf, lg, B, cols)
        out = pd.DataFrame({"타자": roster["batter"].map(bn).values, "타석": st_, "예측 wOBA": P2 @ WOBA_W,
                            "출루 확률": P2[:, 1] + P2[:, 2] + P2[:, 3] + P2[:, 4], "삼진 확률": P2[:, 0], "홈런 확률": P2[:, 4]})
        out = out.sort_values("예측 wOBA", ascending=False).head(9).reset_index(drop=True)
        out.index += 1
        for c in ("출루 확률", "삼진 확률", "홈런 확률"):
            out[c] *= 100
        st.dataframe(out, column_config={"예측 wOBA": st.column_config.NumberColumn(format="%.3f"),
                                         **{c: st.column_config.NumberColumn(format="%.1f%%") for c in ("출루 확률", "삼진 확률", "홈런 확률")}})
