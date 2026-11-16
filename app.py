"""
惜食預報 FoodCast — 程式進入點

啟動方式：streamlit run app.py
這個檔案只負責設定頁面與左側選單，各頁內容放在 pages/ 資料夾。
"""
import streamlit as st

st.set_page_config(page_title="惜食預報 FoodCast", page_icon="🍱", layout="wide")

nav = st.navigation([
    st.Page("pages/home.py", title="首頁", icon="🏠", default=True),
    st.Page("pages/student.py", title="學生人潮預報", icon="🎓"),
    st.Page("pages/store.py", title="店家備餐建議", icon="🧑‍🍳"),
    st.Page("pages/analysis.py", title="模型分析", icon="📊"),
    st.Page("pages/assistant.py", title="AI 助理", icon="🤖"),
])
nav.run()
