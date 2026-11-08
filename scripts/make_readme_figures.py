"""README용 그림 3개를 data/processed 결과로 다시 그립니다.

    python scripts/make_readme_figures.py

- images/abs_bb_monthly.png        월별 볼넷률과 ABS 효과
- images/luck_persistence.png      운과 실력 지표의 다음 시즌 지속성
- images/luck_regression_check.png 유의하게 운이 좋았던 타자의 다음 시즌 wOBA
"""
from pathlib import Path
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT=Path(__file__).resolve().parents[1]
R=str(ROOT/"data/processed")+"/"; OUT=str(ROOT/"images")+"/"
(ROOT/"images").mkdir(exist_ok=True)
# 한글 폰트: 설치된 CJK 폰트를 찾아 쓰고, 없으면 기본 폰트로 그립니다
_cands=[f for f in fm.findSystemFonts() if any(k in f for k in ("NotoSansCJK","NanumGothic","AppleSDGothicNeo","malgun"))]
if _cands:
    fm.fontManager.addfont(_cands[0]); plt.rcParams["font.family"]=fm.FontProperties(fname=_cands[0]).get_name()
S="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; GRID="#e4e3df"; C1="#2a78d6"; C2="#eb6834"; C3="#1baf7a"
plt.rcParams.update({"figure.facecolor":S,"axes.facecolor":S,"axes.edgecolor":GRID,"axes.labelcolor":T2,
 "xtick.color":T2,"ytick.color":T2,"text.color":T1,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True})
# A: ABS monthly BB%
m=pd.read_parquet(R+"abs_monthly_rates.parquet").sort_values(["season","month"])
m["lab"]=m.season.astype(str)+"-"+m.month.astype(str).str.zfill(2); x=np.arange(len(m)); y=m["BB%"]*100
fig,ax=plt.subplots(figsize=(10,4))
pre=m.season<2026
ax.plot(x[pre],y[pre],color=C1,lw=2,marker="o",ms=4,label="2021–2025")
ax.plot(x[~pre],y[~pre],color=C2,lw=2,marker="o",ms=5,label="2026 (ABS 챌린지 도입)")
first=m.groupby("season").head(1).index
pos=[list(m.index).index(i) for i in first]
ax.set_xticks(pos,[str(s) for s in m.loc[first,"season"]])
ax.set_ylabel("볼넷률 BB% (%)"); ax.set_title("월별 볼넷률: 2026년 ABS 챌린지 도입 이후 상승",loc="left",fontsize=13)
ax.legend(frameon=False,loc="upper left")
its=pd.read_parquet(R+"abs_its.parquet").set_index("지표")
ax.text(0.99,0.04,f"추세·계절성 통제 후 ABS 효과 +{its.loc['BB%','ABS 효과(%p)']:.2f}%p (p<0.001)",transform=ax.transAxes,ha="right",color=T2,fontsize=10)
fig.tight_layout(); fig.savefig(OUT+"abs_bb_monthly.png",dpi=150); plt.close()
# B: persistence
meta=json.load(open(R+"luck_meta.json"))
groups=["투수","타자"]; luck=[meta["era_resid_persistence"],meta["luck_persistence_b"]]; skill=[meta["k_persistence"],meta["skill_persistence_b_xwoba"]]
fig,ax=plt.subplots(figsize=(8,4)); xi=np.arange(2); w=0.34
b1=ax.bar(xi-w/2-0.01,luck,w,color=C2,label="성적과 과정의 차이 (운)")
b2=ax.bar(xi+w/2+0.01,skill,w,color=C1,label="과정 지표 (실력)")
for bars in (b1,b2):
    for r in bars: ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.015,f"{r.get_height():.2f}",ha="center",color=T1,fontsize=11)
ax.set_xticks(xi,["투수\n(ERA−기대 ERA  vs  삼진율)","타자\n(wOBA−xwOBA  vs  xwOBA)"])
ax.set_ylim(0,0.8); ax.set_ylabel("올해와 다음 시즌의 상관")
ax.set_title("운은 다음 시즌으로 이어지지 않고, 실력은 이어진다",loc="left",fontsize=13); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT+"luck_persistence.png",dpi=150); plt.close()
# C: regression of significant lucky batters
b=pd.read_parquet(R+"batter_season.parquet")
rows=[]
for n,s in [("Tyler Fitzgerald",2024),("Randy Arozarena",2021),("TJ Friedl",2023)]:
    a=b[(b.name==n)&(b.season==s)].iloc[0]; c=b[(b.name==n)&(b.season==s+1)].iloc[0]
    rows.append((f"{n}\n({s}→{s+1})",a.wOBA,a.xwOBA,c.wOBA))
fig,ax=plt.subplots(figsize=(9,4)); xi=np.arange(3); w=0.26
sets=[("해당 시즌 wOBA (실제)",1,C2),("해당 시즌 xwOBA (기대)",2,C1),("다음 시즌 wOBA (실제)",3,C3)]
for k,(lab,j,col) in enumerate(sets):
    v=[r[j] for r in rows]; bars=ax.bar(xi+(k-1)*(w+0.02),v,w,color=col,label=lab)
    for r in bars: ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.004,f"{r.get_height():.3f}",ha="center",fontsize=9,color=T1)
ax.set_xticks(xi,[r[0] for r in rows]); ax.set_ylim(0,0.42); ax.set_ylabel("wOBA")
ax.set_title("통계적으로 '운이 좋았던' 타자 3명은 모두 다음 시즌 성적이 내려갔다",loc="left",fontsize=13)
ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.18))
fig.tight_layout(); fig.savefig(OUT+"luck_regression_check.png",dpi=150); plt.close()
print("images/ 에 README 그림 3개를 저장했습니다.")
