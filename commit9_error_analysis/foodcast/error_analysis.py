"""
錯誤分析：模型在什麼情況下最容易猜錯？

整體的 MAE、R² 只告訴我們「平均」準不準，但店家更在意的是：
「哪些日子不能太相信模型？」所以我們把測試期間的預測誤差，
依照不同情境（星期幾、假日、考試週、下雨……）分組比較，找出模型的弱點。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foodcast import config


def load_errors() -> pd.DataFrame:
    """合併測試期間的預測結果與原始特徵，並計算誤差。"""
    pred = pd.read_csv(config.MODEL_DIR / "crowd_test_pred.csv", parse_dates=["date"])
    feats = pd.read_csv(config.CROWD_CSV, parse_dates=["date"])
    df = pred.merge(feats.drop(columns="customers"), on=["date", "meal", "slot"], how="left")
    df["error"] = df["pred"] - df["customers"]          # 正數 = 高估（備太多），負數 = 低估
    df["abs_error"] = df["error"].abs()
    return df


def _scenario(df: pd.DataFrame) -> pd.Series:
    """把每一列歸到一個主要情境（依優先順序，只取一個，方便比較）。"""
    conds = [
        df["in_semester"] == 0,
        df["is_holiday"] == 1,
        df["is_weekend"] == 1,
        df["exam_week"] == 1,
        df["pre_final"] == 1,
        df["rain_mm"] > 0,
        df["weekday"] == 4,
    ]
    names = ["寒暑假", "國定假日", "週末", "考試週", "期末考前一週", "下雨的平日", "週五"]
    return pd.Series(np.select(conds, names, default="一般平日"), index=df.index)


def by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """各情境的平均誤差、相對誤差與偏差方向。"""
    d = df.assign(情境=_scenario(df))
    g = d.groupby("情境").agg(
        筆數=("error", "size"),
        平均人數=("customers", "mean"),
        MAE=("abs_error", "mean"),
        偏差=("error", "mean"),
    )
    # 相對誤差：MAE 佔平均人數的比例。人少的情境 MAE 自然小，所以要看比例才公平
    g["相對誤差 (%)"] = 100 * g["MAE"] / g["平均人數"]
    g = g.round({"平均人數": 0, "MAE": 1, "偏差": 1, "相對誤差 (%)": 1})
    return g.sort_values("相對誤差 (%)", ascending=False).reset_index()


def by_slot(df: pd.DataFrame) -> pd.DataFrame:
    """各時段的 MAE，看看尖峰時段是不是比較難預測。"""
    g = df.groupby(["meal", "slot"]).agg(MAE=("abs_error", "mean"), 平均人數=("customers", "mean"))
    return g.round(1).reset_index()


def worst_days(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """整天誤差最大的幾天，並標出當天的特殊狀況。"""
    day = df.groupby("date").agg(
        實際人數=("customers", "sum"), 預測人數=("pred", "sum"), 誤差合計=("abs_error", "sum"),
        rain_mm=("rain_mm", "first"), exam_week=("exam_week", "first"), is_holiday=("is_holiday", "first"),
        in_semester=("in_semester", "first"), weekday=("weekday", "first"), has_event=("has_event", "first"),
    )

    def note(r) -> str:
        tags = []
        if not r["in_semester"]:
            tags.append("寒暑假")
        if r["is_holiday"]:
            tags.append("國定假日")
        if r["exam_week"]:
            tags.append("考試週")
        if r["has_event"]:
            tags.append("校園活動")
        if r["rain_mm"] > 0:
            tags.append(f"下雨 {r['rain_mm']:.0f}mm")
        return "、".join(tags) or "—"

    day["狀況"] = day.apply(note, axis=1)
    day["星期"] = day["weekday"].map(lambda w: "週" + "一二三四五六日"[int(w)])
    out = day.sort_values("誤差合計", ascending=False).head(n).reset_index()
    out["日期"] = out["date"].dt.strftime("%Y/%m/%d")
    return out[["日期", "星期", "實際人數", "預測人數", "誤差合計", "狀況"]].astype(
        {"預測人數": int, "誤差合計": int})
