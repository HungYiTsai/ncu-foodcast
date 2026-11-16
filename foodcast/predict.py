"""
預測服務：給定日期與天氣，產生人潮預報與備餐建議。介面 (Streamlit) 只呼叫這裡的函式。
"""
from __future__ import annotations

import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd

from foodcast import config, train
from foodcast.calendar_tw import day_features
from foodcast.data_generator import MONTH_TEMP
from foodcast.features import crowd_X, demand_X

# 學餐座位數：每 30 分鐘時段大約可以服務的人數，用來換算擁擠程度
SEATS_PER_SLOT = 380
CROWD_LEVELS = [(0.45, "舒適"), (0.75, "普通"), (1.0, "擁擠"), (np.inf, "爆滿")]


def ensure_models() -> None:
    """模型檔不存在時自動訓練，確保第一次 Demo 也能直接開啟。"""
    if not (config.CROWD_MODEL.exists() and config.DEMAND_MODELS.exists() and config.METRICS_JSON.exists()):
        train.main()


def load_models() -> tuple[object, dict]:
    ensure_models()
    return joblib.load(config.CROWD_MODEL), joblib.load(config.DEMAND_MODELS)


def load_metrics() -> dict:
    ensure_models()
    return json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))


def default_weather(d: dt.date) -> dict:
    """沒有天氣預報時，用該月份的平均氣溫、不下雨當預設值。"""
    return {"temp": float(MONTH_TEMP[d.month]), "rain_mm": 0.0}


def crowd_level(customers: float) -> str:
    ratio = customers / SEATS_PER_SLOT
    return next(label for limit, label in CROWD_LEVELS if ratio < limit)


def _base_row(d: dt.date, temp: float, rain_mm: float, has_event: int) -> dict:
    return {"date": pd.Timestamp(d), "temp": temp, "rain_mm": rain_mm,
            "has_event": int(has_event), **day_features(d)}


def forecast_crowd(model, d: dt.date, temp: float, rain_mm: float, has_event: int = 0) -> pd.DataFrame:
    """預測某一天每個時段的來客數與擁擠程度。"""
    base = _base_row(d, temp, rain_mm, has_event)
    rows = [{**base, "meal": meal, "slot": slot}
            for meal in config.MEALS for slot in config.SLOTS[meal]]
    df = pd.DataFrame(rows)
    df["customers"] = model.predict(crowd_X(df)).clip(min=0).round().astype(int)
    df["level"] = df["customers"].map(crowd_level)
    return df[["meal", "slot", "customers", "level"]]


def best_slot(crowd_df: pd.DataFrame, meal: str) -> pd.Series:
    """找出某一餐人最少的時段，給學生「什麼時候去最不用排隊」的建議。"""
    sub = crowd_df[crowd_df["meal"] == meal]
    return sub.loc[sub["customers"].idxmin()]


def forecast_demand(models: dict, d: dt.date, temp: float, rain_mm: float,
                    has_event: int = 0, promo_dish: str | None = None,
                    quantile: float = config.DEFAULT_QUANTILE) -> pd.DataFrame:
    """
    預測某一天各餐各品項的需求，回傳：
      expected   預期需求（中位數 q=0.5）
      recommend  建議備餐量（第 quantile 分位數，無條件進位）
      buffer     建議多備的份數 = recommend - expected
    """
    base = _base_row(d, temp, rain_mm, has_event)
    rows = [{**base, "meal": meal, "dish": dish, "is_promo": int(dish == promo_dish)}
            for meal in config.MEALS for dish in config.DISH_NAMES]
    df = pd.DataFrame(rows)
    X = demand_X(df)
    df["expected"] = np.round(models[0.5].predict(X).clip(min=0)).astype(int)
    df["recommend"] = np.ceil(models[quantile].predict(X).clip(min=0)).astype(int)
    # 分位數模型各自訓練，極少數情況高分位數可能略低於中位數，這裡強制修正
    df["recommend"] = np.maximum(df["recommend"], df["expected"])
    df["buffer"] = df["recommend"] - df["expected"]
    return df[["meal", "dish", "expected", "recommend", "buffer"]]


def day_summary(d: dt.date, crowd_df: pd.DataFrame, demand_df: pd.DataFrame,
                temp: float, rain_mm: float) -> dict:
    """把預測結果整理成精簡的字典，給 AI 助理當作回答依據。"""
    f = day_features(d)
    tags = []
    if not f["in_semester"]:
        tags.append("寒暑假")
    if f["is_holiday"]:
        tags.append("國定假日")
    if f["exam_week"]:
        tags.append("考試週")
    if f["pre_final"]:
        tags.append("期末考前一週")
    if f["is_weekend"]:
        tags.append("週末")
    summary = {
        "date": d.isoformat(),
        "weekday": "一二三四五六日"[d.weekday()],
        "tags": tags,
        "weather": {"temp": temp, "rain_mm": rain_mm},
        "meals": {},
    }
    for meal in config.MEALS:
        c = crowd_df[crowd_df["meal"] == meal]
        dm = demand_df[demand_df["meal"] == meal]
        peak = c.loc[c["customers"].idxmax()]
        quiet = best_slot(crowd_df, meal)
        summary["meals"][meal] = {
            "total_customers": int(c["customers"].sum()),
            "peak_slot": peak["slot"], "peak_level": peak["level"],
            "quiet_slot": quiet["slot"], "quiet_level": quiet["level"],
            "slots": dict(zip(c["slot"], c["customers"].astype(int))),
            "dishes": {r.dish: {"expected": int(r.expected), "recommend": int(r.recommend)}
                       for r in dm.itertuples()},
        }
    return summary
