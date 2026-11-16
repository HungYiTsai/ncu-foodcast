"""首頁：專案簡介與回測成果摘要。"""
import streamlit as st

from foodcast import config
from foodcast.ui import APP_NAME, get_metrics, get_models

get_models()  # 先載入（或第一次訓練）模型
m = get_metrics()
bt = m["demand"]["backtest"]

st.title(f"🍱 {APP_NAME}")
st.subheader("用 AI 預測學餐人潮與需求：學生少排隊、店家少剩食")

st.markdown(
    """
**問題**：學餐午餐尖峰大排長龍，店家又只能憑經驗備餐。遇到下雨、考試週、連假前後，
常常不是賣完，就是剩下一大堆，最後變成廚餘。

**我們的做法**：用行事曆（考試週、假日）與天氣資料訓練機器學習模型：
- 🎓 **學生端**：預測每 30 分鐘的人潮，告訴你幾點去最不用排隊
- 🧑‍🍳 **店家端**：預測每個品項的需求，給出「建議備餐量」，並且可以自己調整剩食和缺貨之間的取捨
- 🤖 **AI 助理**：直接用中文問「明天中午幾點去人最少？」
"""
)

st.markdown(f"#### 回測成果（{bt['test_days']} 天測試期間，服務水準 q={config.DEFAULT_QUANTILE}）")
c1, c2, c3, c4 = st.columns(4)
c1.metric("剩食減少", f"{bt['waste_reduction_pct']}%",
          f"{bt['current']['waste_kg'] - bt['ai']['waste_kg']:,.0f} kg")
c2.metric("需求滿足率", f"{bt['ai']['fill_rate']:.1%}",
          f"{(bt['ai']['fill_rate'] - bt['current']['fill_rate']) * 100:+.1f} 個百分點")
c3.metric("估計減碳", f"{bt['co2e_saved_kg'] / 1000:,.1f} 噸 CO₂e")
best = m["crowd"]["best_model"]
c4.metric("人潮預測 R²", f"{m['crowd']['comparison'][best]['R2']:.3f}", best, delta_color="off")

st.info(
    "⚠️ 目前使用**模擬資料**（依行事曆、天氣與點餐習慣的假設產生），用來驗證整套流程。"
    "取得學餐實際銷售資料後，可直接替換並重新訓練，詳見 README。",
)
st.caption("👈 從左側選單開始使用")
