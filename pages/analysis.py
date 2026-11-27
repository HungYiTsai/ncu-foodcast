"""模型分析：模型比較、特徵重要性、預測 vs 實際、每日剩食回測。給評審與 Q&A 用。"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from foodcast import config
from foodcast.ui import get_metrics, get_models

get_models()
m = get_metrics()

st.title("📊 模型分析")
st.caption(f"訓練期間：{m['train_period']}｜測試期間：{m['test_period']}"
           "（依時間切分，模型沒看過測試期間的資料）")

# ---- 1. 人潮模型比較 ----
st.subheader("1. 人潮模型比較（每 30 分鐘來客數）")
comp = pd.DataFrame(m["crowd"]["comparison"]).T.reset_index(names="模型")
c1, c2 = st.columns([1, 1])
c1.dataframe(comp.drop(columns="選用", errors="ignore"), hide_index=True, width="stretch")
c1.markdown(f"選用 **{m['crowd']['best_model']}**：MAE 最低。線性迴歸表現比基準線差，"
            "因為人潮和時段、星期幾是**非線性**關係（例如 12:00 尖峰、週五晚餐驟降）。")
comp["選用"] = comp["模型"].eq(m["crowd"]["best_model"]).map({True: "選用模型", False: "其他"})
fig = px.bar(comp, x="模型", y="MAE", text="MAE", color="選用",
             color_discrete_map={"選用模型": "#2E7D9E", "其他": "#B8C4CC"},
             labels={"MAE": "平均絕對誤差 MAE（人）"})
fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
c2.plotly_chart(fig, width="stretch")

# ---- 2. 特徵重要性 ----
st.subheader("2. 哪些因素最影響人潮？（置換重要性）")
imp = pd.DataFrame(m["crowd"]["importance"])
fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
             labels={"importance": "打亂後 MAE 增加量", "feature": ""})
fig.update_layout(height=420, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")

# ---- 3. 預測 vs 實際 ----
st.subheader("3. 預測 vs 實際（測試期間任選一週）")
ct = pd.read_csv(config.MODEL_DIR / "crowd_test_pred.csv", parse_dates=["date"])
weeks = sorted(ct["date"].dt.to_period("W").astype(str).unique())
wk = st.select_slider("選擇週次", options=weeks, value=weeks[min(7, len(weeks) - 1)])
sub = ct[ct["date"].dt.to_period("W").astype(str) == wk].copy()
sub["時間"] = sub["date"].dt.strftime("%m/%d ") + sub["slot"]
long = sub.melt(id_vars="時間", value_vars=["customers", "pred"], var_name="類型", value_name="人數")
long["類型"] = long["類型"].map({"customers": "實際", "pred": "預測"})
fig = px.line(long, x="時間", y="人數", color="類型", markers=True)
fig.update_layout(height=360, margin=dict(t=10), xaxis_tickangle=-60)
st.plotly_chart(fig, width="stretch")

# ---- 4. 備餐回測 ----
st.subheader("4. 每日剩食回測：目前方式 vs AI 建議")
bt = pd.read_csv(config.MODEL_DIR / "demand_backtest.csv", parse_dates=["date"])
qcol = f"prep_q{int(config.DEFAULT_QUANTILE * 100)}"
kg = bt["dish"].map({k: v["waste_kg"] for k, v in config.DISHES.items()})
bt["目前方式"] = (bt["prep_current"] - bt["demand"]).clip(lower=0) * kg
bt["AI 建議"] = (bt[qcol] - bt["demand"]).clip(lower=0) * kg
daily = bt.groupby("date")[["目前方式", "AI 建議"]].sum().reset_index()
fig = px.line(daily, x="date", y=["目前方式", "AI 建議"],
              labels={"value": "剩食 (kg)", "date": "", "variable": ""},
              color_discrete_sequence=["#C8423B", "#2E7D9E"])
fig.update_layout(height=360, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("目前方式參考「前兩週」銷量，遇到連假、考試週、天氣變化時反應慢，剩食尖峰明顯；"
           "AI 模型事先知道行事曆與天氣，所以能提前調整。")

pa = m["demand"]["point_accuracy"]
st.markdown(f"品項需求模型（中位數）測試集：MAE **{pa['MAE']} 份**、R² **{pa['R2']}**")

# ---- 5. 超參數調整（時間序列交叉驗證）----
st.subheader("5. 超參數調整：時間序列交叉驗證")
tuning_path = config.MODEL_DIR / "tuning.json"
if not tuning_path.exists():
    st.info("尚未執行超參數調整。在終端機執行 `python -m foodcast.tuning`（約 1～3 分鐘）後重新整理本頁。")
else:
    import json

    tu = json.loads(tuning_path.read_text(encoding="utf-8"))
    res = pd.DataFrame(tu["results"])
    base = tu["baseline_params"]
    is_base = ((res["learning_rate"] == base["learning_rate"]) & (res["max_leaf_nodes"] == base["max_leaf_nodes"])
               & (res["min_samples_leaf"] == base["min_samples_leaf"]))
    res["標記"] = np.where(res.index == 0, "CV 最佳", np.where(is_base, "原本設定", ""))
    st.caption(f"共 {tu['n_combinations']} 組超參數 × {tu['n_splits']} folds；只用訓練期間資料，"
               "並以「日期」為單位切分，永遠用較早的日子訓練、較晚的日子驗證。")

    c1, c2 = st.columns([1, 1])
    show = res.head(10).rename(columns={"cv_mae": "CV MAE", "cv_std": "CV 標準差"})
    c1.dataframe(show, hide_index=True, width="stretch")
    fig = px.scatter(res, x="learning_rate", y="cv_mae", color=res["max_leaf_nodes"].astype(str),
                     size=res["min_samples_leaf"], error_y="cv_std",
                     labels={"learning_rate": "學習率", "cv_mae": "CV MAE（人）", "color": "max_leaf_nodes"})
    fig.update_layout(height=340, margin=dict(t=10))
    c2.plotly_chart(fig, width="stretch")

    before, after = tu["test_mae"]["原本設定"], tu["test_mae"]["調整後設定"]
    st.markdown(f"**測試集 MAE**：原本設定 {before} → CV 最佳設定 {after}")
    if after < before:
        st.success("調整後在測試集也變好，可以考慮採用新的超參數。")
    else:
        gap = float(res.loc[is_base, "cv_mae"].iloc[0] - res["cv_mae"].iloc[0]) if is_base.any() else 0.0
        st.warning(f"CV 最佳的設定在測試集**沒有比較好**：它和原本設定的 CV 差距只有 {gap:.2f} 人，"
                   f"比各 fold 的標準差（約 {res['cv_std'].iloc[0]:.1f} 人）還小，代表差異多半是隨機波動。"
                   "因此我們**維持原本設定**，避免對驗證資料過度調參。")
