"""
學生回報：讓學生回報「現在實際排隊狀況」，並和模型預測比較。

用途：
  1. 驗證模型：預測「舒適」但學生回報「大排長龍」，代表模型可能有問題。
  2. 累積真實資料：回報會存成 CSV，未來可以當作新的訓練資料（資料回饋循環）。

目前回報存在執行網頁那台電腦的 data/feedback.csv（不會上傳 GitHub）。
正式上線時可以改存到雲端資料庫，讓所有學生的回報集中在一起。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from foodcast import config

FEEDBACK_CSV = config.DATA_DIR / "feedback.csv"

# 學生看得懂的排隊描述 → 對應到預測的擁擠程度
QUEUE_OPTIONS = {
    "幾乎不用排": "舒適",
    "排一下（5 分鐘內）": "普通",
    "要排一陣子（5～15 分鐘）": "擁擠",
    "大排長龍（15 分鐘以上）": "爆滿",
}
LEVEL_ORDER = ["舒適", "普通", "擁擠", "爆滿"]
COLUMNS = ["reported_at", "date", "meal", "slot", "reported_level", "predicted_level", "predicted_customers"]


def nearest_slot(now: dt.datetime) -> tuple[str, str]:
    """依照現在時間，找出最接近的用餐時段，作為表單的預設值。"""
    minutes = now.hour * 60 + now.minute
    best, best_gap = ("午餐", "12:00"), 10 ** 9
    for meal in config.MEALS:
        for slot in config.SLOTS[meal]:
            h, m = map(int, slot.split(":"))
            gap = abs(h * 60 + m + 15 - minutes)      # 時段中間點
            if gap < best_gap:
                best, best_gap = (meal, slot), gap
    return best


def save(d: dt.date, meal: str, slot: str, queue_option: str,
         predicted_level: str, predicted_customers: int,
         path=FEEDBACK_CSV) -> dict:
    """新增一筆回報。"""
    rec = {
        "reported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": d.isoformat(), "meal": meal, "slot": slot,
        "reported_level": QUEUE_OPTIONS[queue_option],
        "predicted_level": predicted_level,
        "predicted_customers": int(predicted_customers),
    }
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame([rec], columns=COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    return rec


def load(path=FEEDBACK_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, encoding="utf-8-sig")


def agreement(df: pd.DataFrame) -> dict:
    """
    預測與回報的一致程度：
      exact   完全一致的比例
      within1 差距在一級以內的比例（例如預測「普通」、回報「擁擠」）
      bias    平均差幾級（正數 = 實際比預測更擠，模型低估）
    """
    if df.empty:
        return {"n": 0, "exact": None, "within1": None, "bias": None}
    rep = df["reported_level"].map(LEVEL_ORDER.index)
    pred = df["predicted_level"].map(LEVEL_ORDER.index)
    diff = rep - pred
    return {
        "n": int(len(df)),
        "exact": round(float((diff == 0).mean()), 3),
        "within1": round(float((diff.abs() <= 1).mean()), 3),
        "bias": round(float(diff.mean()), 2),
    }
