"""學生端：預測各時段人潮，推薦最不用排隊的時間，並顯示未來 7 天熱度圖。"""
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from foodcast import config, predict
from foodcast.ui import LEVEL_COLORS, get_models, sidebar_inputs

crowd_model, _ = get_models()
day, temp, rain, event = sidebar_inputs()

st.title("🎓 學生人潮預報")
st.caption(f"{day:%Y/%m/%d}（週{'一二三四五六日'[day.weekday()]}）｜氣溫 {temp}°C｜雨量 {rain} mm")

crowd = predict.forecast_crowd(crowd_model, day, temp, rain, event)

# ---- 推薦卡片 ----
cols = st.columns(2)
for col, meal in zip(cols, config.MEALS):
    quiet = predict.best_slot(crowd, meal)
    sub = crowd[crowd["meal"] == meal]
    peak = sub.loc[sub["customers"].idxmax()]
    col.success(f"**{meal}推薦 {quiet['slot']} 前往**（預計 {quiet['customers']} 人，{quiet['level']}）\n\n"
                f"尖峰是 {peak['slot']}（約 {peak['customers']} 人，{peak['level']}）")

# ---- 各時段長條圖 ----
fig = px.bar(crowd, x="slot", y="customers", color="level", facet_col="meal",
             color_discrete_map=LEVEL_COLORS, category_orders={"level": list(LEVEL_COLORS),
                               "slot": config.SLOTS["午餐"] + config.SLOTS["晚餐"]},
             labels={"slot": "時段", "customers": "預測人數", "level": "擁擠程度", "meal": ""})
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.add_hline(y=predict.SEATS_PER_SLOT, line_dash="dot", annotation_text="座位容量", annotation_position="top left")
fig.update_xaxes(matches=None)
fig.update_layout(height=380, margin=dict(t=40, b=10))
st.plotly_chart(fig, width="stretch")

# ---- 未來 7 天熱度圖 ----
st.subheader("未來 7 天人潮熱度")
rows = []
for i in range(7):
    d = day + dt.timedelta(days=i)
    w = predict.default_weather(d) if i else {"temp": temp, "rain_mm": rain}
    c = predict.forecast_crowd(crowd_model, d, w["temp"], w["rain_mm"], event if i == 0 else 0)
    c["day"] = f"{d:%m/%d} 週{'一二三四五六日'[d.weekday()]}"
    rows.append(c)
week = pd.concat(rows)
week["時段"] = week["meal"] + " " + week["slot"]
heat = week.pivot(index="day", columns="時段", values="customers")
heat = heat[[f"{m} {s}" for m in config.MEALS for s in config.SLOTS[m]]]
fig2 = px.imshow(heat, color_continuous_scale="YlOrRd", aspect="auto", text_auto=True,
                 labels={"color": "預測人數", "x": "", "y": ""})
fig2.update_layout(height=360, margin=dict(t=10, b=10))
st.plotly_chart(fig2, width="stretch")
st.caption("第 2 天之後使用該月平均氣溫、不下雨的預設天氣。")
