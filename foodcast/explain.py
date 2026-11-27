"""
預測原因解釋：告訴使用者「為什麼這一天人比較多／比較少」。

方法：反事實比較 (counterfactual)
  對每一個「特殊因素」（下雨、考試週、假日……），把它改回「一般狀態」再預測一次，
  兩次預測的差距，就是這個因素造成的影響。例如：
      實際預測（下雨 15 mm）：午餐 1,320 人
      假設沒下雨          ：午餐 1,180 人
      → 下雨讓人數增加約 140 人（+12%）

  這種做法不用打開模型內部，任何模型都適用（model-agnostic），
  而且說明方式很直覺，適合給一般使用者看。
  注意：各因素是「分別」比較，彼此之間可能有交互作用，所以加總不一定等於總差距。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from foodcast import config
from foodcast.calendar_tw import day_features
from foodcast.data_generator import MONTH_TEMP
from foodcast.features import crowd_X


def _meal_total(model, base: dict, meal: str) -> float:
    """某一餐所有時段的預測人數加總。"""
    rows = [{**base, "meal": meal, "slot": s} for s in config.SLOTS[meal]]
    return float(model.predict(crowd_X(pd.DataFrame(rows))).clip(min=0).sum())


def _factors(base: dict, d: dt.date) -> list[tuple[str, dict]]:
    """
    找出這一天有哪些「特殊因素」，回傳 (因素名稱, 改回一般狀態的修改內容)。
    只列出當天確實存在的因素。
    """
    out = []
    if base["rain_mm"] > 0:
        out.append((f"下雨（{base['rain_mm']:.0f} mm）", {"rain_mm": 0.0}))
    if base["exam_week"]:
        out.append(("考試週", {"exam_week": 0}))
    if base["pre_final"]:
        out.append(("期末考前一週", {"pre_final": 0}))
    if base["has_event"]:
        out.append(("校園活動", {"has_event": 0}))
    if base["is_holiday"]:
        out.append(("國定假日", {"is_holiday": 0}))
    if not base["in_semester"]:
        out.append(("寒暑假", {"in_semester": 1, "week": 8}))
    elif base["week"] in (1, 2):
        out.append(("開學前兩週", {"week": 8}))
    if base["weekday"] >= 4:   # 週五、週六、週日和平日差最多，以週三當作一般狀態
        label = "週" + "一二三四五六日"[base["weekday"]]
        out.append((label, {"weekday": 2, "is_weekend": 0}))
    normal_temp = MONTH_TEMP[d.month]
    if abs(base["temp"] - normal_temp) >= 3:
        word = "偏熱" if base["temp"] > normal_temp else "偏冷"
        out.append((f"氣溫{word}（{base['temp']:.0f}°C）", {"temp": float(normal_temp)}))
    return out


def explain_crowd(model, d: dt.date, temp: float, rain_mm: float,
                  has_event: int = 0) -> pd.DataFrame:
    """
    回傳每一餐、每個特殊因素的影響：
      meal      午餐/晚餐
      factor    因素名稱
      effect    這個因素造成的人數變化（正數 = 增加）
      pct       變化百分比（相對於「沒有這個因素」時的人數）
    沒有任何特殊因素時回傳空表。
    """
    base = {"temp": temp, "rain_mm": rain_mm, "has_event": int(has_event), **day_features(d)}
    rows = []
    for meal in config.MEALS:
        actual = _meal_total(model, base, meal)
        for name, change in _factors(base, d):
            without = _meal_total(model, {**base, **change}, meal)
            effect = actual - without
            rows.append({
                "meal": meal, "factor": name,
                "effect": round(effect),
                "pct": round(100 * effect / without, 1) if without > 0 else 0.0,
            })
    return pd.DataFrame(rows, columns=["meal", "factor", "effect", "pct"])


def explain_sentence(df: pd.DataFrame, meal: str, top: int = 3) -> str:
    """把影響最大的幾個因素整理成一句話，例如「下雨 +12%、考試週 +18%」。"""
    sub = df[df["meal"] == meal]
    sub = sub[sub["pct"].abs() >= 2]                          # 影響小於 2% 的不提
    sub = sub.loc[sub["pct"].abs().sort_values(ascending=False).index]
    if sub.empty:
        return "一般日，沒有特別影響人潮的因素。"
    parts = [f"{r.factor} {r.pct:+.0f}%" for r in sub.head(top).itertuples()]
    return "主要影響：" + "、".join(parts)
