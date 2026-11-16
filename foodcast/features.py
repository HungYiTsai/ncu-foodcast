"""
特徵工程：把原始資料表轉成模型的輸入矩陣 X。

訓練（train.py）與預測（predict.py）共用同一個函式，
確保兩邊的特徵欄位、順序、編碼方式完全一致（避免 training-serving skew）。
"""
from __future__ import annotations

import pandas as pd

from foodcast import config

# 行事曆 + 天氣特徵（兩個模型共用）
BASE_FEATURES = [
    "meal_code", "weekday", "is_weekend", "is_holiday", "month",
    "in_semester", "week", "exam_week", "pre_final",
    "temp", "rain_mm", "has_event",
]
CROWD_FEATURES = BASE_FEATURES + ["slot_hour"]
DEMAND_FEATURES = BASE_FEATURES + ["dish_code", "is_promo"]

# 特徵中文名稱，用於介面上的特徵重要性圖
FEATURE_LABELS = {
    "meal_code": "午餐/晚餐", "weekday": "星期幾", "is_weekend": "是否週末",
    "is_holiday": "國定假日", "month": "月份", "in_semester": "是否學期中",
    "week": "學期週次", "exam_week": "考試週", "pre_final": "期末考前一週",
    "temp": "氣溫", "rain_mm": "雨量", "has_event": "校園活動",
    "slot_hour": "用餐時段", "dish_code": "品項", "is_promo": "今日特餐",
}


def _slot_to_hour(slot: str) -> float:
    """'12:30' → 12.5"""
    h, m = slot.split(":")
    return int(h) + int(m) / 60


def _common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["meal_code"] = (out["meal"] == "晚餐").astype(int)
    return out


def crowd_X(df: pd.DataFrame) -> pd.DataFrame:
    """人潮模型的特徵矩陣。"""
    out = _common(df)
    out["slot_hour"] = out["slot"].map(_slot_to_hour)
    return out[CROWD_FEATURES]


def demand_X(df: pd.DataFrame) -> pd.DataFrame:
    """品項需求模型的特徵矩陣。品項用整數編碼，並在模型中標記為類別特徵。"""
    out = _common(df)
    out["dish_code"] = out["dish"].map({d: i for i, d in enumerate(config.DISH_NAMES)})
    if "is_promo" not in out:
        out["is_promo"] = 0
    return out[DEMAND_FEATURES]


# 告訴 HistGradientBoosting 哪一欄是類別特徵（品項沒有大小順序）
DEMAND_CATEGORICAL = [DEMAND_FEATURES.index("dish_code")]
