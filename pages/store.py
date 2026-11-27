"""店家端：各品項建議備餐量，以及「剩食 vs 缺貨」的服務水準取捨。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from foodcast import config, predict
from foodcast.ui import get_metrics, get_models, sidebar_inputs

_, demand_models = get_models()
metrics = get_metrics()
day, temp, rain, event = sidebar_inputs()

st.title("🧑‍🍳 店家備餐建議")
st.caption(f"{day:%Y/%m/%d}（週{'一二三四五六日'[day.weekday()]}）｜氣溫 {temp}°C｜雨量 {rain} mm")

c1, c2 = st.columns(2)
q = c1.select_slider(
    "服務水準（數字越高越不容易賣完，但剩食越多）",
    options=config.QUANTILES, value=config.DEFAULT_QUANTILE,
    format_func=lambda x: f"{int(x * 100)}%",
)
promo = c2.selectbox("今日特餐", ["無"] + config.DISH_NAMES)
promo = None if promo == "無" else promo

df = predict.forecast_demand(demand_models, day, temp, rain, event, promo, q)

# ---- 建議表 ----
for meal in config.MEALS:
    sub = df[df["meal"] == meal].drop(columns="meal").rename(
        columns={"dish": "品項", "expected": "預期需求（份）", "recommend": "建議備餐（份）", "buffer": "安全量（份）"})
    st.markdown(f"#### {meal}　合計建議 **{sub['建議備餐（份）'].sum()} 份**")
    st.dataframe(sub, hide_index=True, width="stretch")

st.download_button("⬇️ 下載備餐建議 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"備餐建議_{day}.csv", mime="text/csv")

# ---- 未來 7 天備餐計畫（方便店家一次規劃一週的進貨）----
st.subheader("📅 未來 7 天備餐計畫")
plan = predict.weekly_plan(demand_models, day, q, first_day_weather={"temp": temp, "rain_mm": rain})
pivot = plan.pivot_table(index="dish", columns="日期", values="recommend", aggfunc="sum")
pivot = pivot[plan["日期"].unique()]          # 依日期順序排列欄位
pivot.loc["合計"] = pivot.sum()
st.dataframe(pivot, width="stretch")
st.caption("單位：份（午餐＋晚餐合計）。第 2 天之後使用該月平均氣溫、不下雨的預設天氣，"
           "實際進貨前請參考氣象預報調整。")
st.download_button("⬇️ 下載 7 天備餐計畫 CSV", plan.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"7天備餐計畫_{day}.csv", mime="text/csv")

# ---- 取捨曲線 ----
st.subheader("剩食與缺貨的取捨（回測結果）")
t = pd.DataFrame(metrics["demand"]["tradeoff"])
cur = metrics["demand"]["backtest"]["current"]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t["waste_kg"], y=t["fill_rate"] * 100, mode="lines+markers+text", name="AI 建議",
    text=[f"{int(x * 100)}%" for x in t["quantile"]], textposition="top left",
    marker=dict(size=[16 if x == q else 9 for x in t["quantile"]], color="#2E7D9E")))
fig.add_trace(go.Scatter(
    x=[cur["waste_kg"]], y=[cur["fill_rate"] * 100], mode="markers+text", name="目前備餐方式",
    text=["目前方式"], textposition="bottom left", marker=dict(size=14, color="#C8423B", symbol="x")))
fig.update_layout(xaxis_title="測試期間總剩食 (kg)", yaxis_title="需求滿足率 (%)",
                  height=420, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("每個藍點都在紅色 ✕ 的左上方：AI 建議在任何服務水準下，都比目前方式剩食更少、滿足率更高或相近。"
           "目前方式 = 前兩週同一天銷量 × 1.15。")
