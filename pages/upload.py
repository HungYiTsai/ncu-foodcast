"""資料上傳：店家上傳真實銷售／人潮資料，檢查格式後重新訓練模型。"""
import pandas as pd
import plotly.express as px
import streamlit as st

from foodcast import data_io, train

st.title("📤 上傳真實資料")
st.markdown(
    "目前模型使用**模擬資料**。學餐提供真實的 POS 銷售紀錄或人潮計數後，可以在這裡上傳，"
    "系統會自動檢查格式、補上行事曆特徵（星期幾、考試週、假日），再重新訓練模型。"
)

kind_label = st.radio("資料類型", ["品項銷售量（備餐模型）", "時段人潮（人潮模型）"], horizontal=True)
kind = "demand" if kind_label.startswith("品項") else "crowd"

with st.expander("📋 欄位說明與範本"):
    req = "、".join(f"`{c}`" for c in data_io.REQUIRED[kind])
    st.markdown(f"**必填欄位**：{req}　｜　**選填**：`temp`、`rain_mm`、`has_event`"
                + ("、`is_promo`、`prep_current`（實際備餐量，用來計算目前剩食）" if kind == "demand" else ""))
    tpl = data_io.template(kind)
    st.dataframe(tpl, hide_index=True, width="stretch")
    st.download_button("⬇️ 下載範本 CSV", tpl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"template_{kind}.csv", mime="text/csv")

file = st.file_uploader("選擇 CSV 檔", type="csv")
if file is not None:
    try:
        raw = pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        raw = pd.read_csv(file, encoding="cp950")   # Excel 存成 CSV 時常見的 Big5 編碼

    df, messages = data_io.validate(raw, kind)
    for msg in messages:
        (st.error if msg.startswith("❌") else st.warning if msg.startswith("⚠️") else st.info)(msg)

    if df is not None:
        target = data_io.TARGET[kind]
        c1, c2, c3 = st.columns(3)
        c1.metric("資料筆數", f"{len(df):,}")
        c2.metric("天數", df["date"].nunique())
        c3.metric("期間", f"{df['date'].min():%Y/%m/%d} ~ {df['date'].max():%Y/%m/%d}")

        daily = df.groupby("date")[target].sum().reset_index()
        fig = px.line(daily, x="date", y=target, labels={"date": "", target: "每日合計"})
        fig.update_layout(height=280, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.head(20), hide_index=True, width="stretch")

        st.warning("按下按鈕會用這份資料**取代**目前的資料並重新訓練。原本的模擬資料會自動備份，可隨時還原。")
        if st.button("✅ 使用這份資料重新訓練", type="primary"):
            data_io.save(df, kind)
            with st.spinner("重新訓練中（約 30 秒）…"):
                train.main()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("完成！其他頁面已改用新模型。")

st.divider()
if st.button("↩️ 還原成模擬資料並重新訓練"):
    if data_io.restore_simulated():
        with st.spinner("重新訓練中…"):
            train.main()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("已還原成模擬資料。")
    else:
        st.info("目前就是模擬資料，不需要還原。")
