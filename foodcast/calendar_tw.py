"""
簡化版校園行事曆：從日期推算學期週次、考試週、寒暑假與國定假日。

注意：這是依「每學期 18 週、第 9 週期中考、第 18 週期末考」的規則推算的
模擬行事曆，不是學校官方行事曆。若取得正式行事曆，替換 `academic_info()`
裡的規則即可，其他模組都不需要改。
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

# 固定日期的國定假日（月, 日）
FIXED_HOLIDAYS = {(1, 1), (2, 28), (4, 4), (4, 5), (5, 1), (9, 28), (10, 10), (10, 25), (12, 25)}

# 農曆節日（端午、中秋）換算成國曆的日期；春節落在寒假中，不另外列
LUNAR_HOLIDAYS = {
    dt.date(2024, 9, 17), dt.date(2025, 5, 31), dt.date(2025, 10, 6),
    dt.date(2026, 6, 19), dt.date(2026, 9, 25), dt.date(2027, 6, 9), dt.date(2027, 9, 15),
}


def _first_monday_on_or_after(d: dt.date) -> dt.date:
    """回傳 d 當天或之後的第一個星期一。"""
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


@lru_cache(maxsize=16)
def semester_starts(year: int) -> tuple[dt.date, dt.date]:
    """
    回傳某學年度的 (上學期開學日, 下學期開學日)。
    year 為上學期所在的西元年，例如 2025 代表 114 學年度。
    規則：上學期 = 9/7 之後第一個週一；下學期 = 隔年 2/17 之後第一個週一。
    """
    fall = _first_monday_on_or_after(dt.date(year, 9, 7))
    spring = _first_monday_on_or_after(dt.date(year + 1, 2, 17))
    return fall, spring


def is_holiday(d: dt.date) -> bool:
    return (d.month, d.day) in FIXED_HOLIDAYS or d in LUNAR_HOLIDAYS


def academic_info(d: dt.date) -> dict:
    """
    回傳日期 d 的校園狀態：
      in_semester  是否在學期中（非寒暑假）
      week         學期第幾週（1~18，假期為 0）
      exam_week    是否為期中或期末考週
      pre_final    是否為期末考前一週（讀書人潮高峰）
    """
    # 先判斷 d 屬於哪個學年度
    year = d.year if d.month >= 8 else d.year - 1
    fall, spring = semester_starts(year)

    for start in (fall, spring):
        delta = (d - start).days
        if 0 <= delta < 18 * 7:
            week = delta // 7 + 1
            return {
                "in_semester": 1,
                "week": week,
                "exam_week": int(week in (9, 18)),
                "pre_final": int(week == 17),
            }
    return {"in_semester": 0, "week": 0, "exam_week": 0, "pre_final": 0}


def day_features(d: dt.date) -> dict:
    """把一個日期轉成模型可用的行事曆特徵。"""
    info = academic_info(d)
    return {
        "weekday": d.weekday(),          # 0=週一 ... 6=週日
        "is_weekend": int(d.weekday() >= 5),
        "is_holiday": int(is_holiday(d)),
        "month": d.month,
        **info,
    }
