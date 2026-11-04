import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from lib import CAT, DIV, INK, INK2, MUTED, PCFG, SEQ, SURFACE, load, meta, names, note, radar, require_data, setup

setup("🧭 투수·타자 유형")
require_data()

kind_lab = st.radio("대상", ["투수", "타자"], horizontal=True)
GROUP_COLOR = {"포심 계열": CAT[0], "싱커 계열": CAT[1], "커터 계열": CAT[2]}


def gcolor(g):
    return GROUP_COLOR.get(g, CAT[3] if "2" in str(g) else MUTED)


# =====================================================================================  투수
if kind_lab == "투수":
    pm = meta("ptype_meta")
    T = load("pitcher_types")
    if T.empty:
        st.info("투수 유형 결과가 없습니다.")
        st.stop()
    groups = load("ptype_groups").set_index("유형")["상위 그룹"].to_dict()
    order = [t for t in ["포심-슬라이더형", "포심-스위퍼형", "포심-커브형", "포심-체인지업형", "포심-스플리터형",
                         "싱커-슬라이더형", "싱커-커브형", "싱커-체인지업형", "커터형"] if t in set(T["유형"])]
    st.markdown("투수 유형은 **가장 많이 던지는 패스트볼(포심·싱커·커터) × 가장 많이 던지는 보조구종** 조합으로 정의하고, "
                "강속구·하이 포심·땅볼형·피치 구성·역할은 태그로 붙였습니다. 아래 분석으로 이 유형들이 **실제로 구분되는지, 성적이 다른지** 검증합니다.")
    view = st.segmented_control("보기", ["유형 분포", "계층적 군집", "판별분석(LDA)", "성적 차이 검정", "대응분석", "유형 전환", "SHAP(구위)"],
                                default="유형 분포", key="ptype_view") or "유형 분포"

    if view == "유형 분포":
        season = st.selectbox("시즌", sorted(T["season"].unique(), reverse=True))
        d = T[T["season"] == season]
        cnt = d["유형"].value_counts().reindex(order).fillna(0)
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Bar(y=cnt.index, x=cnt.values, orientation="h",
                                   marker=dict(color=[gcolor(groups.get(t)) for t in cnt.index], line=dict(color=SURFACE, width=2)),
                                   text=[f"{int(v)}명" for v in cnt.values], textposition="outside",
                                   hovertemplate="%{y}: %{x}명<extra></extra>"))
            fig.update_layout(title=f"{season} 유형별 투수 수 (색 = 상위 그룹)", height=420, yaxis=dict(autorange="reversed"),
                              xaxis_title="투수 수", margin=dict(l=120))
            st.plotly_chart(fig, width="stretch", config=PCFG)
            st.markdown(" ".join(f"<span class='badge' style='border-left:4px solid {c}'>{g}</span>" for g, c in GROUP_COLOR.items()),
                        unsafe_allow_html=True)
        with c2:
            ps = load("pitcher_season")
            dd = d.merge(ps[["pitcher", "season", "K%", "pitches"]], on=["pitcher", "season"], how="left")
            rows = []
            for t in order:
                x = dd[dd["유형"] == t].sort_values("pitches", ascending=False)
                if x.empty:
                    continue
                rows.append({"유형": t, "상위 그룹": groups.get(t), "투수 수": len(x), "K%": x["K%"].mean() * 100,
                             "땅볼%": x["GB%"].mean() * 100, "패스트볼 구속": x["패스트볼 평균구속"].mean(),
                             "강속구 비율": x["태그"].str.contains("강속구").mean() * 100,
                             "대표 투수": ", ".join(x["name"].head(3))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=380, column_config={
                **{c: st.column_config.NumberColumn(format="%.1f") for c in ("K%", "땅볼%", "패스트볼 구속", "강속구 비율")}})
        st.markdown("##### 유형 비율의 시즌별 변화")
        tr = load("ptype_trend")
        pick = st.multiselect("비교할 유형 (최대 4개)", order, default=[t for t in ("포심-슬라이더형", "포심-스위퍼형") if t in order],
                              max_selections=4)
        fig = go.Figure()
        for i, t in enumerate(pick):
            x = tr[tr["유형"] == t]
            fig.add_trace(go.Scatter(x=x["season"], y=x["비율"] * 100, name=t, mode="lines+markers",
                                     line=dict(color=CAT[i], width=2), marker=dict(size=8)))
        fig.update_layout(height=320, yaxis_title="규정 투수 중 비율 (%)", xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch", config=PCFG)
        note("규정 표본 = 시즌 300구 이상. 선수는 시즌마다 유형이 바뀔 수 있습니다('유형 전환' 보기).")

    elif view == "계층적 군집":
        st.markdown("9개 유형의 평균 성향(삼진·볼넷·헛스윙·땅볼·강한 타구·구속·무브먼트 등 11개 지표)으로 유형 간 거리를 재고, "
                    "**Ward 연결법**으로 가까운 유형부터 묶었습니다. 덴드로그램에서 가장 큰 높이 간격을 기준으로 상위 그룹을 정했습니다.")
        dd = load("ptype_dendrogram")
        prof = load("ptype_profile").set_index("유형")
        if len(dd):
            leaves = list(dd["ivl"].iloc[0])
            fig = go.Figure()
            for ic, dc in zip(dd["icoord"], dd["dcoord"]):
                fig.add_trace(go.Scatter(x=list(ic), y=list(dc), mode="lines", line=dict(color=INK2, width=2), hoverinfo="skip", showlegend=False))
            xs = [5 + 10 * i for i in range(len(leaves))]
            fig.add_trace(go.Scatter(x=xs, y=[0] * len(xs), mode="markers", marker=dict(size=14, color=[gcolor(groups.get(t)) for t in leaves],
                                     line=dict(color=SURFACE, width=2)), hovertext=leaves, hoverinfo="text", showlegend=False))
            fig.update_layout(title=f"유형 덴드로그램 → 상위 그룹 {pm.get('k_groups')}개", height=420,
                              xaxis=dict(tickvals=xs, ticktext=leaves, tickangle=-30), yaxis_title="Ward 거리", margin=dict(b=110))
            st.plotly_chart(fig, width="stretch", config=PCFG)
            gtab = pd.DataFrame({"상위 그룹": sorted(set(groups.values()))})
            gtab["포함 유형"] = gtab["상위 그룹"].map(lambda g: ", ".join(t for t in leaves if groups.get(t) == g))
            st.dataframe(gtab, hide_index=True)
            note("데이터로 묶어도 패스트볼 종류(포심·싱커·커터)별로 갈라지면, '어떤 패스트볼을 주로 던지는가'가 투수 성향을 가르는 1차 기준이라는 뜻입니다.")
            P = prof.reindex(leaves)
            fig = go.Figure(go.Heatmap(z=P.values, x=list(P.columns), y=leaves, colorscale=DIV, zmid=0, zmin=-1.5, zmax=1.5, xgap=2, ygap=2,
                                       text=np.vectorize(lambda v: f"{v:.2f}")(P.values.astype(float)), texttemplate="%{text}", textfont=dict(size=10),
                                       colorbar=dict(title="Z", thickness=10), hovertemplate="%{y} · %{x}: Z %{z:.2f}<extra></extra>"))
            fig.update_layout(title="유형별 평균 (리그 대비 Z점수, 덴드로그램 순서)", height=90 + 38 * len(leaves), xaxis=dict(tickangle=-30),
                              margin=dict(l=120))
            st.plotly_chart(fig, width="stretch", config=PCFG)

    elif view == "판별분석(LDA)":
        lm = pm.get("lda", {})
        st.markdown("**구종 비율은 빼고 공의 물리 특성만**(패스트볼과 보조구종의 구속·회전·무브먼트, 릴리스 위치) 넣어 **선형판별분석(LDA)** 으로 "
                    "유형을 맞혀봤습니다. 같은 투수가 학습·검증에 동시에 들어가지 않도록 투수 단위 5겹 교차검증을 했습니다.")
        c = st.columns(3)
        c[0].metric("교차검증 정확도", f"{lm.get('accuracy', 0) * 100:.1f}%")
        c[1].metric("기준선 (가장 많은 유형으로 다 찍기)", f"{lm.get('baseline', 0) * 100:.1f}%")
        ari = pm.get("ari_vs_kmeans")
        c[2].metric("K-means 군집과의 일치도 (ARI)", "–" if ari is None else f"{ari:.2f}")
        L = load("ptype_lda")
        c1, c2 = st.columns([3, 2])
        with c1:
            season = st.selectbox("시즌", sorted(L["season"].unique(), reverse=True), key="lda_s")
            x = L[L["season"] == season]
            fig = go.Figure()
            for g in sorted(x["상위 그룹"].dropna().unique()):
                y = x[x["상위 그룹"] == g]
                fig.add_trace(go.Scattergl(x=y["ld1"], y=y["ld2"], mode="markers", name=g,
                                           marker=dict(size=7, color=gcolor(g), opacity=.55, line=dict(color=SURFACE, width=1)),
                                           customdata=np.c_[y["name"], y["유형"]],
                                           hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
            cen = x.groupby("유형")[["ld1", "ld2"]].mean()
            fig.add_trace(go.Scatter(x=cen["ld1"], y=cen["ld2"], mode="text", text=cen.index, textfont=dict(color=INK, size=11),
                                     hoverinfo="skip", showlegend=False))
            ex = lm.get("explained", [0, 0])
            fig.update_layout(title=f"판별 지도 (판별축 설명력 {ex[0] * 100:.0f}% + {ex[1] * 100:.0f}%)", height=520,
                              xaxis_title="판별축 1", yaxis_title="판별축 2")
            st.plotly_chart(fig, width="stretch", config=PCFG)
        if True:
            conf = load("ptype_confusion").set_index("실제")
            fig = go.Figure(go.Heatmap(z=conf.values * 100, x=list(conf.columns), y=list(conf.index), colorscale=SEQ, zmin=0, zmax=100,
                                       text=np.round(conf.values * 100).astype(int), texttemplate="%{text}", textfont=dict(size=10),
                                       xgap=1, ygap=1, colorbar=dict(title="%", thickness=10),
                                       hovertemplate="실제 %{y} → 예측 %{x}: %{z:.0f}%<extra></extra>"))
            fig.update_layout(title="혼동행렬 (행 기준 %, 대각선 = 맞힌 비율)", height=480, xaxis=dict(tickangle=-30, title="예측"),
                              yaxis=dict(autorange="reversed", title="실제"), margin=dict(l=110, b=110))
            st.plotly_chart(fig, width="stretch", config=PCFG)
        rc = load("ptype_lda_recall").sort_values("정확도(재현율)", ascending=False)
        c2.markdown("##### 유형별 맞힌 비율")
        c2.dataframe(rc, hide_index=True, height=380, column_config={"정확도(재현율)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        note("정확도가 높은 유형은 공의 물리적 성질부터 다른 '실재하는' 유형이고, 서로 많이 헷갈리는 유형은 이름은 달라도 공의 성질이 비슷하다는 뜻입니다. "
             "K-means와의 ARI가 낮다는 것은, 숫자만으로 자동 군집하면 야구적 유형이 잘 드러나지 않는다는 뜻이기도 합니다.")

    elif view == "성적 차이 검정":
        st.markdown("9개 유형의 성적이 정말 다른지 **Kruskal-Wallis 검정**(순위 기반, 분포 가정 없음)으로 보고, 어떤 유형끼리 다른지 "
                    "**Dunn 사후검정 + FDR 보정**으로 확인했습니다. 같은 투수가 여러 시즌 들어가면 독립성이 깨지므로 **투수별 가장 최근 시즌 1개**만 썼습니다.")
        kw = load("ptype_kw")
        st.dataframe(kw, hide_index=True, column_config={"H": st.column_config.NumberColumn(format="%.1f"),
                                                          "p": st.column_config.NumberColumn(format="%.4f"),
                                                          "q_FDR": st.column_config.NumberColumn(format="%.4f"),
                                                          "효과크기 ε²": st.column_config.NumberColumn(format="%.3f")})
        note("ε²: 유형이 지표 차이를 얼마나 설명하는지 (0.01 작음 · 0.06 중간 · 0.14 큼).")
        ci = load("ptype_ci")
        m = st.selectbox("지표", list(kw["지표"]))
        x = ci[ci["지표"] == m].sort_values("평균")
        pct = m not in ("xwOBA", "RV/100")
        k = 100 if pct else 1
        fig = go.Figure(go.Scatter(x=x["평균"] * k, y=x["유형"], mode="markers",
                                   marker=dict(size=12, color=[gcolor(groups.get(t)) for t in x["유형"]], line=dict(color=SURFACE, width=2)),
                                   error_x=dict(type="data", symmetric=False, array=(x["상한"] - x["평균"]) * k,
                                                arrayminus=(x["평균"] - x["하한"]) * k, color=MUTED, thickness=2),
                                   customdata=x["n"], hovertemplate="%{y}: %{x:.3f} (n=%{customdata})<extra></extra>"))
        fig.update_layout(title=f"유형별 {m} 평균과 95% 신뢰구간", height=420, xaxis_title=m + (" (%)" if pct else ""), margin=dict(l=120))
        st.plotly_chart(fig, width="stretch", config=PCFG)
        dn = load("ptype_dunn")
        dn = dn[dn["지표"] == m]
        labs = [t for t in order if t in set(dn["유형1"]) | set(dn["유형2"])]
        M = pd.DataFrame(np.nan, index=labs, columns=labs)
        for _, r in dn.iterrows():
            M.loc[r["유형1"], r["유형2"]] = M.loc[r["유형2"], r["유형1"]] = r["q_FDR"]
        txt = M.map(lambda v: "" if pd.isna(v) else ("***" if v < .001 else "**" if v < .01 else "*" if v < .05 else "·"))
        fig = go.Figure(go.Heatmap(z=-np.log10(M.values.astype(float)), x=labs, y=labs, colorscale=SEQ, zmin=0, zmax=4, xgap=1, ygap=1,
                                   text=txt.values, texttemplate="%{text}", textfont=dict(color=INK),
                                   colorbar=dict(title="-log10 q", thickness=10),
                                   customdata=M.values, hovertemplate="%{y} vs %{x}<br>q = %{customdata:.4f}<extra></extra>"))
        fig.update_layout(title=f"{m} 유형 쌍별 Dunn 검정 (진할수록 차이가 확실, * q<.05 ** q<.01 *** q<.001)", height=460,
                          xaxis=dict(tickangle=-35), yaxis=dict(autorange="reversed"), margin=dict(l=120, b=110))
        st.plotly_chart(fig, width="stretch", config=PCFG)

    elif view == "대응분석":
        ca = load("ptype_ca")
        cam = pm.get("ca", {})
        if ca.empty:
            st.info("대응분석 결과가 없습니다.")
        else:
            st.markdown("유형 9개 × 타석 결과 7개 교차표를 **대응분석(Correspondence Analysis)** 으로 한 지도에 그렸습니다. "
                        "유형 점과 결과 점이 **같은 방향에 있을수록** 그 유형에서 그 결과가 평균보다 많이 나옵니다.")
            c = st.columns(3)
            ish = cam.get("inertia_share", [0, 0])
            c[0].metric("2차원 설명력", f"{(ish[0] + ish[1]) * 100:.0f}%")
            c[1].metric("독립성 χ² 검정", f"p = {cam.get('chi2_p', 1):.1e}")
            c[2].metric("연관 강도 (Cramér's V)", f"{cam.get('cramers_v', 0):.3f}")
            r, o = ca[ca["구분"] == "유형"], ca[ca["구분"] == "결과"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r["dim1"], y=r["dim2"], mode="markers+text", text=r["이름"], textposition="top center", name="투수 유형",
                                     marker=dict(size=12, color=[gcolor(groups.get(t)) for t in r["이름"]], line=dict(color=SURFACE, width=2)),
                                     textfont=dict(color=INK2, size=11)))
            fig.add_trace(go.Scatter(x=o["dim1"], y=o["dim2"], mode="markers+text", text=o["이름"], textposition="bottom center", name="타석 결과",
                                     marker=dict(size=13, symbol="diamond", color=INK, line=dict(color=SURFACE, width=2)),
                                     textfont=dict(color=INK, size=12)))
            fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1))
            fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
            fig.update_layout(title="유형 × 타석 결과 대응분석 지도", height=560, xaxis_title=f"차원 1 ({ish[0] * 100:.0f}%)",
                              yaxis_title=f"차원 2 ({ish[1] * 100:.0f}%)")
            st.plotly_chart(fig, width="stretch", config=PCFG)
            sh = load("ptype_outcome_share").set_index("유형")
            dev = (sh - sh.mean()) * 100
            fig = go.Figure(go.Heatmap(z=dev.values, x=list(dev.columns), y=list(dev.index), colorscale=DIV, zmid=0, xgap=2, ygap=2,
                                       text=np.vectorize(lambda v: f"{v * 100:.1f}")(sh.values.astype(float)), texttemplate="%{text}%", textfont=dict(size=10),
                                       colorbar=dict(title="평균 대비 %p", thickness=10),
                                       hovertemplate="%{y} · %{x}: 평균 대비 %{z:+.1f}%p<extra></extra>"))
            fig.update_layout(title="유형별 타석 결과 비율 (숫자 = 실제 비율, 색 = 유형 평균 대비 차이)", height=420, margin=dict(l=120))
            st.plotly_chart(fig, width="stretch", config=PCFG)
            note(f"분석 타석 {cam.get('n_pa', 0):,}개. 타석 결과는 원래 우연이 커서 Cramér's V는 작게 나오지만, 표본이 커서 유형 간 차이 자체는 통계적으로 확실합니다.")

    elif view == "유형 전환":
        tm = pm.get("transition", {})
        st.markdown("같은 투수의 유형이 다음 시즌에 어떻게 바뀌었는지 추적하고, **유형을 바꾼 투수와 유지한 투수의 다음 시즌 변화**를 비교했습니다. "
                    "성적이 극단적이었던 투수는 다음 해 평균으로 돌아가는 경향(평균 회귀)이 있어, **직전 시즌 값과 나이를 통제한 회귀(HC1)** 로 효과를 추정했습니다.")
        c = st.columns(2)
        c[0].metric("다음 시즌에 유형이 바뀐 비율", f"{tm.get('switch_rate', 0) * 100:.0f}%")
        sw = tm.get("sweeper", {})
        if sw.get("K%"):
            c[1].metric("포심-슬라이더 → 포심-스위퍼 전환 효과 (K%)", f"{sw['K%']['효과'] * 100:+.2f}%p",
                        f"p = {sw['K%']['p']:.2f}, n = {sw['K%']['n']}", delta_color="off")
        fl = load("ptype_flow")
        fl = fl[fl["n"] >= 5]
        labs = order
        idx = {t: i for i, t in enumerate(labs)}
        fig = go.Figure(go.Sankey(
            node=dict(label=[f"{t}" for t in labs] + [f"{t} " for t in labs], pad=10, thickness=14,
                      color=[gcolor(groups.get(t)) for t in labs] * 2, line=dict(color=SURFACE, width=1)),
            link=dict(source=[idx[a] for a in fl["유형"]], target=[len(labs) + idx[b] for b in fl["유형_다음"]], value=fl["n"].tolist(),
                      color=["rgba(195,194,183,0.35)" if a == b else "rgba(42,120,214,0.35)" for a, b in zip(fl["유형"], fl["유형_다음"])],
                      hovertemplate="%{source.label} → %{target.label}: %{value}명<extra></extra>")))
        fig.update_layout(title="시즌 → 다음 시즌 유형 흐름 (파랑 = 유형 변경, 5명 이상 흐름만)", height=560)
        st.plotly_chart(fig, width="stretch", config=PCFG)
        did = load("ptype_did")
        st.markdown("##### 유형 전환 효과 (이중차분형 비교)")
        st.dataframe(did, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in did.columns if c not in ("지표", "전환 n", "유지 n")})
        note("전환 효과 = 직전 값·나이가 같은 투수끼리 비교했을 때, 유형을 바꾼 투수의 다음 시즌 변화가 유지한 투수보다 얼마나 더 컸는지. "
             "p가 크면 '유형을 바꾸는 것 자체로는 성적이 달라지지 않았다'는 뜻입니다.")

    elif view == "SHAP(구위)":
        sm = meta("shap_meta").get("stuff", {})
        sh = load("stuff_shap")
        st.markdown("LightGBM으로 **구종의 물리 특성만으로 헛스윙률**을 예측하고, SHAP으로 각 특성의 기여를 봤습니다.")
        c = st.columns(3)
        c[0].metric("투수 단위 5겹 CV R²", f"{sm.get('cv_r2', float('nan')):.3f}")
        if sm.get("stuff_to_next_whiff_r") is not None:
            c[1].metric("구위 지수 → 내년 헛스윙률 r", f"{sm['stuff_to_next_whiff_r']:.2f}")
            c[2].metric("올해 헛스윙률 → 내년 헛스윙률 r", f"{sm['whiff_to_next_whiff_r']:.2f}")
        note("구위 지수는 실제 헛스윙률보다 내년 예측력이 낮습니다. 따라서 '헛스윙률보다 나은 지표'가 아니라, "
             "표본이 적을 때(시즌 초반, 새로 장착한 구종) 공의 성질만으로 헛스윙률을 미리 추정하는 용도로 해석해야 합니다.")
        imp = pd.Series(sm.get("importance", {})).sort_values()
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2))))
            fig.update_layout(title="평균 |SHAP|", height=60 + 26 * len(imp), margin=dict(l=130))
            st.plotly_chart(fig, width="stretch", config=PCFG)
        with c2:
            if len(imp):
                f = st.selectbox("특성별 SHAP 의존도", imp.index[::-1])
                fig = go.Figure(go.Scattergl(x=sh[f"val_{f}"], y=sh[f"shap_{f}"], mode="markers", marker=dict(size=5, color=CAT[0], opacity=.35)))
                fig.add_hline(y=0, line=dict(color=MUTED, width=1))
                fig.update_layout(height=380, xaxis_title=f, yaxis_title="SHAP 값 (헛스윙률 기여)")
                st.plotly_chart(fig, width="stretch", config=PCFG)

# =====================================================================================  타자
else:
    cm = meta("clusters_meta").get("batter", {})
    cl = load("batter_clusters")
    view = st.segmented_control("보기", ["유형 지도", "유형별 특징", "SHAP(타자)"], default="유형 지도", key="btype_view") or "유형 지도"
    types = sorted(cl["유형"].unique()) if len(cl) else []
    color = {t: CAT[i % 8] for i, t in enumerate(types)}
    if view == "유형 지도":
        season = st.selectbox("시즌", ["전체"] + sorted(cl["season"].unique(), reverse=True))
        d = cl if season == "전체" else cl[cl["season"] == season]
        d = d.assign(name=d["batter"].map(names("batter")))
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for t in types:
                x = d[d["유형"] == t]
                fig.add_trace(go.Scattergl(x=x["pc1"], y=x["pc2"], mode="markers", name=f"유형 {t} ({len(x)})",
                                           marker=dict(size=7, color=color[t], line=dict(color=SURFACE, width=1), opacity=.8),
                                           customdata=np.c_[x["name"], x["season"]],
                                           hovertemplate="%{customdata[0]} (%{customdata[1]})<extra>유형 " + t + "</extra>"))
            pv = cm.get("pca_var", [0, 0])
            fig.update_layout(title=f"PCA 2차원 지도 (설명력 {pv[0] * 100:.0f}% + {pv[1] * 100:.0f}%)", height=500,
                              xaxis_title="주성분 1", yaxis_title="주성분 2")
            st.plotly_chart(fig, width="stretch", config=PCFG)
        with c2:
            ks = load("batter_k_selection")
            st.markdown(f"##### k = {cm.get('k')} 선택 근거")
            st.dataframe(ks.rename(columns={"inertia": "엘보우(관성)", "silhouette": "실루엣", "davies_bouldin": "Davies-Bouldin", "gmm_bic": "GMM BIC"}),
                         hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("엘보우(관성)", "실루엣", "Davies-Bouldin", "GMM BIC")})
            sil = float(ks.loc[ks["k"] == cm.get("k"), "silhouette"].iloc[0]) if len(ks) else float("nan")
            note(f"실루엣 {sil:.3f}, 부트스트랩 ARI {cm.get('bootstrap_ari_mean', float('nan')):.2f}. 실루엣이 0.25 미만이면 군집 경계가 뚜렷하지 않다는 뜻으로, "
                 "타자 유형은 '뚜렷한 덩어리'라기보다 연속적인 스펙트럼을 편의상 나눈 것으로 해석해야 합니다.")
    elif view == "유형별 특징":
        prof = load("batter_cluster_profile").set_index("유형")
        raw = load("batter_cluster_raw").set_index("유형")
        fig = go.Figure(go.Heatmap(z=prof.values, x=prof.columns, y=[f"유형 {t}" for t in prof.index], colorscale=DIV, zmid=0, zmin=-2, zmax=2,
                                   text=np.vectorize(lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}")(raw.values.astype(float)),
                                   texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2, colorbar=dict(title="Z", thickness=10)))
        fig.update_layout(height=120 + 50 * len(prof), title="유형별 평균 (색 = 리그 대비 Z점수, 숫자 = 실제 평균)", xaxis=dict(tickangle=-35))
        st.plotly_chart(fig, width="stretch", config=PCFG)
        # 레이더: 낮을수록 좋은 지표는 뒤집어 '바깥 = 좋음'으로 통일
        good = {"K%": -1, "Chase%": -1, "Whiff%": -1, "GB%": -1}
        axes = list(prof.columns)
        rs = [(f"유형 {t}", list(norm.cdf((prof.loc[t] * pd.Series({a: good.get(a, 1) for a in axes})).values.astype(float)) * 100), color[t])
              for t in prof.index]
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(radar(rs, [a + ("(낮을수록↑)" if a in good else "") for a in axes], "유형별 레이더 (바깥 = 좋음, 50 = 리그 평균)", height=460),
                        width="stretch", config=PCFG)
        with c2:
            for t in prof.index:
                st.markdown(f"<span class='badge' style='border-left:4px solid {color[t]}'>유형 {t}</span> {cm.get('desc', {}).get(t, '')} · "
                            f"{cm.get('sizes', {}).get(t, 0)}명·시즌", unsafe_allow_html=True)
            note("K%·Chase%·Whiff%·GB%는 낮을수록 좋아서 레이더에서는 뒤집어 그렸습니다. 발사각·당겨치기%처럼 좋고 나쁨이 없는 지표는 값 그대로입니다.")
        pick = st.selectbox("유형 소속 타자 보기", types, format_func=lambda t: f"유형 {t}")
        bs = load("batter_season")
        mem = cl[cl["유형"] == pick].merge(bs[["batter", "season", "name", "PA"]], on=["batter", "season"])
        # 선수당 한 줄: 이 유형으로 분류된 시즌들의 타석 합계 순
        mem = mem.groupby("batter").agg(타자=("name", "last"), PA=("PA", "sum")).sort_values("PA", ascending=False)
        st.dataframe(mem[["타자"]], hide_index=True, height=260)
        note(f"이 유형으로 한 번이라도 분류된 타자 {len(mem)}명 (2021–2026 합산 타석이 많은 순).")
    else:
        sm = meta("shap_meta").get("batter", {})
        sh = load("batter_shap")
        st.markdown("LightGBM으로 **스윙·선구·타구 과정 지표만으로 wOBA**를 예측하고 SHAP으로 기여를 봤습니다. "
                    "K%·BB%는 wOBA 계산에 직접 들어가는 성분이라 입력에서 뺐습니다(정답으로 정답을 설명하는 것을 막기 위해).")
        st.metric("타자 단위 5겹 CV R²", f"{sm.get('cv_r2', float('nan')):.3f}")
        imp = pd.Series(sm.get("importance", {})).sort_values()
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2))))
            fig.update_layout(title="평균 |SHAP|", height=60 + 26 * len(imp), margin=dict(l=130))
            st.plotly_chart(fig, width="stretch", config=PCFG)
        with c2:
            if len(imp):
                f = st.selectbox("특성별 SHAP 의존도", imp.index[::-1])
                fig = go.Figure(go.Scattergl(x=sh[f"val_{f}"], y=sh[f"shap_{f}"], mode="markers", marker=dict(size=5, color=CAT[0], opacity=.35)))
                fig.add_hline(y=0, line=dict(color=MUTED, width=1))
                fig.update_layout(height=380, xaxis_title=f, yaxis_title="SHAP 값 (wOBA 기여)")
                st.plotly_chart(fig, width="stretch", config=PCFG)
