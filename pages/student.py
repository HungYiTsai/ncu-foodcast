"""學生端：預測各時段人潮，推薦最不用排隊的時間，並顯示未來 7 天熱度圖。"""
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from foodcast import config, explain, feedback, predict
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

# ---- 預測原因 ----
reasons = explain.explain_crowd(crowd_model, day, temp, rain, event)
for col, meal in zip(st.columns(2), config.MEALS):
    col.caption("💡 " + explain.explain_sentence(reasons, meal))
if not reasons.empty:
    with st.expander("為什麼會是這個人數？（各因素的影響）"):
        fig_r = px.bar(reasons, x="pct", y="factor", color="meal", barmode="group", orientation="h",
                       text=reasons["pct"].map(lambda v: f"{v:+.0f}%"),
                       labels={"pct": "對人數的影響 (%)", "factor": "", "meal": ""},
                       color_discrete_map={"午餐": "#2E7D9E", "晚餐": "#E07B39"})
        fig_r.add_vline(x=0, line_color="gray")
        fig_r.update_layout(height=120 + 60 * reasons["factor"].nunique(), margin=dict(t=10))
        st.plotly_chart(fig_r, width="stretch")
        st.caption("算法：把每個因素改回「一般狀態」（沒下雨、非考試週、週三……）再預測一次，"
                   "比較兩次的人數差距。各因素分開比較，加總不一定等於總差距。")

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

# ---- 學生回報 ----
st.subheader("📣 回報現在的排隊狀況")
st.caption("幫忙回報實際排隊情形，可以用來檢查預測準不準，也會累積成未來的訓練資料。")
default_meal, default_slot = feedback.nearest_slot(dt.datetime.now())
with st.form("feedback_form", clear_on_submit=True):
    # 表單內的元件要按送出才會更新，所以把「餐別＋時段」合成一個選項，避免時段清單跟不上餐別
    choices = [f"{m} {s}" for m in config.MEALS for s in config.SLOTS[m]]
    c1, c2 = st.columns([1, 3])
    fb_choice = c1.selectbox("時段", choices, index=choices.index(f"{default_meal} {default_slot}"))
    fb_queue = c2.radio("排隊情形", list(feedback.QUEUE_OPTIONS), horizontal=True)
    submitted = st.form_submit_button("送出回報")
    fb_meal, fb_slot = fb_choice.split()
if submitted:
    # 回報的是「今天」的狀況，所以用今天的日期和側邊欄的天氣重新預測該時段
    today = dt.date.today()
    today_crowd = predict.forecast_crowd(crowd_model, today, temp, rain, event)
    row = today_crowd[(today_crowd["meal"] == fb_meal) & (today_crowd["slot"] == fb_slot)].iloc[0]
    rec = feedback.save(today, fb_meal, fb_slot, fb_queue, row["level"], row["customers"])
    same = rec["reported_level"] == rec["predicted_level"]
    st.success(f"感謝回報！你回報「{rec['reported_level']}」，模型預測「{rec['predicted_level']}」"
               + ("，預測正確 👍" if same else "，會作為改進模型的參考。"))

fb = feedback.load()
if not fb.empty:
    ag = feedback.agreement(fb)
    c1, c2, c3 = st.columns(3)
    c1.metric("累積回報", f"{ag['n']} 筆")
    c2.metric("預測完全一致", f"{ag['exact']:.0%}")
    c3.metric("差距在一級以內", f"{ag['within1']:.0%}")
    with st.expander("最近的回報"):
        st.dataframe(fb.tail(10).iloc[::-1], hide_index=True, width="stretch")
