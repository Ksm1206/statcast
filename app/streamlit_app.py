"""Streamlit Cloud 메인 파일: app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="MLB Statcast 해부실", page_icon="⚾", layout="wide")

pages = {
    "개요": [st.Page("pages/home.py", title="프로젝트 개요", icon="🏠", default=True)],
    "선수 해부": [st.Page("pages/pitcher.py", title="투수 리포트", icon="🎯"),
                st.Page("pages/batter.py", title="타자 리포트", icon="🏏")],
    "분석": [st.Page("pages/types.py", title="유형 지도 · SHAP", icon="🧭"),
           st.Page("pages/matchup.py", title="맞대결 시뮬레이터", icon="⚔️"),
           st.Page("pages/composite.py", title="종합 지수", icon="🏆"),
           st.Page("pages/luck.py", title="기대 성적", icon="📈"),
           st.Page("pages/abs.py", title="ABS 전후 비교", icon="🤖"),
           st.Page("pages/method.py", title="검증 · 방법론", icon="🔬")],
}
st.navigation(pages).run()
