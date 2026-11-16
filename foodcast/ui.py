"""
Streamlit 介面共用元件：快取模型、側邊欄輸入、顏色設定。
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from foodcast import predict
from foodcast.calendar_tw import day_features

LEVEL_COLORS = {"舒適": "#2E9E6B", "普通": "#E8B630", "擁擠": "#E07B39", "爆滿": "#C8423B"}
APP_NAME = "惜食預報 FoodCast"


@st.cache_resource(show_spinner="第一次啟動，正在訓練模型（約 30 秒）…")
def get_models():
    """模型只載入一次，之後所有頁面共用（st.cache_resource）。"""
    return predict.load_models()


@st.cache_data
def get_metrics() -> dict:
    return predict.load_metrics()


def sidebar_inputs() -> tuple[dt.date, float, float, int]:
    """側邊欄：選日期與天氣。用 session_state 讓各頁面共用同一組設定。"""
    st.sidebar.header("預測條件")
    default_day = st.session_state.get("day", dt.date.today() + dt.timedelta(days=1))
    day = st.sidebar.date_input("日期", value=default_day, key="day_input")
    st.session_state["day"] = day

    w = predict.default_weather(day)
    if st.session_state.get("temp_month") != day.month:   # 換到別的月份時，重設成該月平均氣溫
        st.session_state["temp"] = w["temp"]
        st.session_state["temp_month"] = day.month
    temp = st.sidebar.slider("預報氣溫 (°C)", 5.0, 38.0, float(st.session_state.get("temp", w["temp"])), 0.5)
    rain = st.sidebar.slider("預報雨量 (mm)", 0.0, 50.0, float(st.session_state.get("rain", 0.0)), 1.0)
    event = st.sidebar.checkbox("當天有校園活動", value=st.session_state.get("event", False))
    st.session_state.update(temp=temp, rain=rain, event=event)

    f = day_features(day)
    tags = []
    if not f["in_semester"]:
        tags.append("寒暑假")
    if f["week"]:
        tags.append(f"學期第 {f['week']} 週")
    if f["exam_week"]:
        tags.append("📝 考試週")
    if f["pre_final"]:
        tags.append("📚 期末考前一週")
    if f["is_holiday"]:
        tags.append("🎌 國定假日")
    st.sidebar.caption("行事曆：" + ("、".join(tags) if tags else "一般日"))
    st.sidebar.caption("※ 天氣預設為該月平均氣溫，可依氣象預報調整。")
    return day, temp, rain, int(event)
