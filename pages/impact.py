"""成效報告：把回測結果換算成省下的錢與減少的碳排，給學校總務處與店家看。"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from foodcast import config, impact
from foodcast.ui import get_models

get_models()   # 確保模型與回測檔已產生
bt = impact.load_backtest()

st.title("💰 成效報告")
q = st.select_slider("服務水準", options=config.QUANTILES, value=config.DEFAULT_QUANTILE,
                     format_func=lambda x: f"{int(x * 100)}%")
s = impact.summary(bt, q)
st.caption(f"回測期間 {bt['date'].min():%Y/%m/%d} ~ {bt['date'].max():%Y/%m/%d}（{s['days']} 天），"
           "比較「目前備餐方式」與「AI 建議」。")

# ---- KPI ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("省下金額", f"NT$ {s['saved_money']:,}", f"每天約 NT$ {s['saved_money_per_day']:,}")
c2.metric("少浪費的餐點", f"{s['saved_servings']:,} 份", f"{s['saved_kg']:,.0f} kg")
c3.metric("估計減碳", f"{s['co2e_kg'] / 1000:,.1f} 噸 CO₂e")
c4.metric("推估一年可省", f"NT$ {s['yearly_money_estimate'] / 10000:,.0f} 萬", "粗估", delta_color="off")

# ---- 損失組成 ----
st.subheader("損失組成：剩食 vs 缺貨")
fig = go.Figure()
for name, key, color in [("剩食損失（食材成本）", "waste_cost", "#C8423B"), ("缺貨損失（少賺的毛利）", "lost_profit", "#E8B630")]:
    fig.add_trace(go.Bar(name=name, x=["目前方式", "AI 建議"], y=[s["current"][key], s["ai"][key]],
                         marker_color=color))
fig.update_layout(barmode="stack", height=340, yaxis_title="NT$", margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("備餐太多會浪費食材，備餐太少會少賺錢。兩種損失都要算，比較才公平。")

# ---- 每月趨勢 ----
st.subheader("每月剩食量")
mon = impact.monthly(bt, q)
fig = px.bar(mon, x="月份", y="waste_kg", color="方式", barmode="group",
             color_discrete_map={"目前方式": "#C8423B", "AI 建議": "#2E7D9E"},
             labels={"waste_kg": "剩食 (kg)"})
fig.update_layout(height=320, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")

# ---- 各品項 ----
st.subheader("各品項成效")
st.dataframe(impact.by_dish(bt, q), hide_index=True, width="stretch")

# ---- 最省錢的服務水準 ----
st.subheader("哪個服務水準最省錢？")
mq = impact.money_by_quantile(bt)
best = mq.loc[mq["總損失"].idxmin()]
fig = px.bar(mq, x="服務水準", y=["剩食損失", "缺貨損失"],
             color_discrete_map={"剩食損失": "#C8423B", "缺貨損失": "#E8B630"},
             labels={"value": "NT$", "variable": ""})
fig.update_layout(height=320, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.info(f"只看金額時，**{best['服務水準']}** 的總損失最低。服務水準越高，剩食損失越多、缺貨損失越少。"
        "不過缺貨也會讓學生不滿意，這是金額算不到的，所以預設採用 70%，"
        "在省錢和讓學生吃得到之間取得平衡。")

with st.expander("計算方式與假設"):
    st.markdown(
        "- **剩食損失** = 剩下的份數 × 每份食材成本\n"
        "- **缺貨損失** = 沒賣到的份數 × 每份毛利（售價 − 成本）\n"
        f"- **減碳** = 剩食公斤數 × {config.CO2E_PER_KG_FOOD_WASTE} kg CO₂e（估計值，僅供呈現量級）\n"
        "- **一年推估** = 回測期間日平均 × 365 天，寒暑假人少，實際可能較低\n"
        "- 各品項的成本、售價為估計值，可在 `foodcast/config.py` 調整\n"
        "- ⚠️ 目前是**模擬資料**的回測結果，換成真實資料後需要重新評估"
    )
    st.dataframe(
        [{"品項": k, "成本 (NT$)": v["cost"], "售價 (NT$)": v["price"], "每份剩食 (kg)": v["waste_kg"]}
         for k, v in config.DISHES.items()],
        hide_index=True, width="stretch",
    )
