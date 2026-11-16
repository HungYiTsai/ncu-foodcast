"""AI 助理：用中文問人潮與備餐問題。有金鑰時用 LLM，沒有時用離線模式。"""
import datetime as dt

import streamlit as st

from foodcast import assistant, predict
from foodcast.ui import get_models, sidebar_inputs

crowd_model, demand_models = get_models()
day, temp, rain, event = sidebar_inputs()

st.title("🤖 AI 助理")
backend = assistant.llm_backend()
st.caption(f"目前模式：{'LLM（' + backend + '）' if backend else '離線模式（未設定 API 金鑰）'}"
           "｜數字一律來自我們訓練的模型，LLM 只負責用口語說明")


def get_summary(d: dt.date) -> dict:
    """產生某一天的預測摘要。選定日期沿用側邊欄天氣，其他日期用該月平均。"""
    w = {"temp": temp, "rain_mm": rain} if d == day else predict.default_weather(d)
    c = predict.forecast_crowd(crowd_model, d, w["temp"], w["rain_mm"], event if d == day else 0)
    dm = predict.forecast_demand(demand_models, d, w["temp"], w["rain_mm"], event if d == day else 0)
    return predict.day_summary(d, c, dm, w["temp"], w["rain_mm"])


if "chat" not in st.session_state:
    st.session_state.chat = []

examples = ["明天中午幾點去人最少？", "晚餐要備幾份？", "週五午餐什麼最熱門？"]
cols = st.columns(len(examples))
pressed = [c.button(q, width="stretch") for c, q in zip(cols, examples)]   # 先畫出全部按鈕
clicked = next((q for q, p in zip(examples, pressed) if p), None)

for role, text in st.session_state.chat:
    st.chat_message(role).markdown(text.replace("\n", "  \n"))

question = st.chat_input("輸入問題，例如：後天晚上會很擠嗎？") or clicked
if question:
    st.chat_message("user").markdown(question)
    # 問題裡的「明天」「週五」以真實今天為準；沒提到日期時用側邊欄選的日期
    reply, mode = assistant.answer(question, dt.date.today(), day, get_summary)
    st.chat_message("assistant").markdown(reply.replace("\n", "  \n"))
    st.session_state.chat += [("user", question), ("assistant", reply)]
