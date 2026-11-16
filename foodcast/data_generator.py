"""
模擬資料產生器：產生兩個學年度的學餐來客數、各品項需求與「目前備餐方式」的剩食。

為什麼用模擬資料？
  目前還沒有取得學餐的 POS 銷售資料，所以先用「有根據的假設」產生資料，
  讓整套流程（前處理 → 訓練 → 評估 → 介面）可以先完整跑通。
  取得真實資料後，只要整理成和 data/crowd.csv、data/demand.csv 相同的欄位，
  就能直接重新訓練，其他程式不用改。

模擬中放入的因果假設（簡報與 Q&A 時可說明）：
  - 午餐 12:00 是尖峰；晚餐 18:00 是尖峰
  - 週五晚餐、週末人潮明顯下降；寒暑假與國定假日大幅下降
  - 考試週與考前一週，晚餐留校人數增加
  - 下雨天少人外出覓食，學餐人數增加
  - 天冷時熱湯麵比例上升，天熱時輕食沙拉比例上升
  - 店家「目前」的備餐方式：參考前兩週同一天的銷量 × 1.15，
    沒有考慮天氣與行事曆，所以假期前後最容易多備而產生剩食

執行：python -m foodcast.data_generator
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from foodcast import config
from foodcast.calendar_tw import day_features

# 台灣北部各月份平均氣溫（°C）與降雨機率，用來模擬天氣
MONTH_TEMP = {1: 15, 2: 16, 3: 18, 4: 22, 5: 25, 6: 28, 7: 29.5, 8: 29, 9: 27, 10: 24, 11: 21, 12: 17}
MONTH_RAIN_P = {1: .35, 2: .40, 3: .40, 4: .40, 5: .50, 6: .55, 7: .30, 8: .35, 9: .35, 10: .25, 11: .30, 12: .30}

# 平日學期中每餐基準來客數
BASE_CUSTOMERS = {"午餐": 1200, "晚餐": 750}
# 每餐各時段的來客比例
SLOT_SHARE = {
    "午餐": [0.10, 0.18, 0.36, 0.24, 0.12],
    "晚餐": [0.12, 0.22, 0.30, 0.22, 0.14],
}
# 星期幾的人潮係數（午餐, 晚餐）
WEEKDAY_FACTOR = {0: (1.00, 1.00), 1: (1.03, 1.02), 2: (1.00, 1.00), 3: (1.02, 1.00),
                  4: (0.88, 0.70), 5: (0.35, 0.30), 6: (0.30, 0.45)}


def simulate_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """產生每日氣溫與雨量。氣溫加入自我相關 (AR(1)) 讓冷熱天數連續出現。"""
    temps, rains = [], []
    anomaly = 0.0
    for d in dates:
        anomaly = 0.7 * anomaly + rng.normal(0, 1.6)        # 寒流/熱浪會持續幾天
        temps.append(round(MONTH_TEMP[d.month] + anomaly, 1))
        rains.append(round(rng.exponential(12), 1) if rng.random() < MONTH_RAIN_P[d.month] else 0.0)
    return pd.DataFrame({"date": dates.date, "temp": temps, "rain_mm": rains})


def meal_multiplier(f: dict, meal: str, temp: float, rain_mm: float, has_event: int) -> float:
    """根據行事曆、天氣、活動計算某一餐的人潮倍率。"""
    lunch_f, dinner_f = WEEKDAY_FACTOR[f["weekday"]]
    m = lunch_f if meal == "午餐" else dinner_f

    if not f["in_semester"]:
        m *= 0.25                                   # 寒暑假
    if f["is_holiday"]:
        m *= 0.35                                   # 國定假日
    if f["exam_week"]:
        m *= 1.05 if meal == "午餐" else 1.18       # 考試週晚餐留校讀書
    if f["pre_final"]:
        m *= 1.08
    if 1 <= f["week"] <= 2:
        m *= 1.08                                   # 開學前兩週新鮮感
    m *= 1 + 0.12 * min(rain_mm / 10, 1.5)          # 下雨天少人外出
    if temp >= 31:
        m *= 0.95
    elif temp <= 13:
        m *= 1.05
    if has_event:
        m *= 1.12                                   # 校園活動
    return m


def dish_shares(temp: float, rain_mm: float, exam_week: int, promo_dish: str | None) -> np.ndarray:
    """根據天氣、考試週與特餐，調整各品項的點餐比例（總和為 1）。"""
    w = {k: v["share"] for k, v in config.DISHES.items()}
    w["麵食"] *= np.clip(1 + 0.035 * (20 - temp), 0.6, 1.6)      # 冷天吃熱湯麵
    w["輕食沙拉"] *= np.clip(1 + 0.045 * (temp - 22), 0.4, 1.7)  # 熱天吃沙拉
    if rain_mm > 0:
        w["麵食"] *= 1.10
    if exam_week:
        w["便當"] *= 1.15                                           # 考試週外帶方便
    if promo_dish:
        w[promo_dish] *= 1.35                                       # 今日特餐
    arr = np.array([w[k] for k in config.DISH_NAMES])
    return arr / arr.sum()


def generate(seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """產生 crowd（時段人潮）與 demand（品項需求/備餐/剩食）兩張表。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(config.SIM_START, config.SIM_END, freq="D")
    weather = simulate_weather(dates, rng).set_index("date")

    crowd_rows, demand_rows = [], []
    for ts in dates:
        d = ts.date()
        f = day_features(d)
        temp, rain = weather.loc[d, "temp"], weather.loc[d, "rain_mm"]
        has_event = int(f["in_semester"] and f["weekday"] < 5 and rng.random() < 0.05)
        promo = rng.choice(config.DISH_NAMES) if rng.random() < 0.12 else None

        for meal in config.MEALS:
            mult = meal_multiplier(f, meal, temp, rain, has_event)
            total = BASE_CUSTOMERS[meal] * mult * rng.lognormal(0, 0.06)

            # 時段人潮：考試週晚餐尖峰往後延（讀完書才去吃）
            shares = np.array(SLOT_SHARE[meal])
            if f["exam_week"] and meal == "晚餐":
                shares = shares * np.array([0.8, 0.9, 1.0, 1.15, 1.25])
            shares = rng.dirichlet(shares / shares.sum() * 300)    # 每天比例略有變動
            for slot, s in zip(config.SLOTS[meal], shares):
                crowd_rows.append({
                    "date": d, "meal": meal, "slot": slot,
                    "customers": int(round(total * s)),
                    "temp": temp, "rain_mm": rain, "has_event": has_event, **f,
                })

            # 品項需求：總人數 × 品項比例，再加上泊松抽樣雜訊
            for dish, s in zip(config.DISH_NAMES, dish_shares(temp, rain, f["exam_week"], promo)):
                demand_rows.append({
                    "date": d, "meal": meal, "dish": dish,
                    "demand": int(rng.poisson(total * s)),
                    "is_promo": int(promo == dish),
                    "temp": temp, "rain_mm": rain, "has_event": has_event, **f,
                })

    crowd = pd.DataFrame(crowd_rows)
    demand = pd.DataFrame(demand_rows)
    demand = add_current_practice(demand)
    return crowd, demand


def add_current_practice(demand: pd.DataFrame) -> pd.DataFrame:
    """
    模擬店家目前的備餐方式：前兩週同一星期幾、同一餐、同一品項的平均銷量 × 1.15，
    四捨五入到 10 份。並計算剩食份數、剩食公斤數與缺貨份數。
    """
    df = demand.sort_values(["dish", "meal", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.weekday
    grp = df.groupby(["dish", "meal", "dow"])["demand"]
    ref = (grp.shift(1) + grp.shift(2)) / 2                 # 前一週、前兩週同一天
    ref = ref.fillna(grp.transform("mean"))                  # 最開始兩週沒有歷史，用平均
    df["prep_current"] = (np.round(ref * 1.15 / 10) * 10).astype(int)

    df["leftover_current"] = (df["prep_current"] - df["demand"]).clip(lower=0)
    df["stockout_current"] = (df["demand"] - df["prep_current"]).clip(lower=0)
    waste_kg = df["dish"].map({k: v["waste_kg"] for k, v in config.DISHES.items()})
    df["waste_kg_current"] = (df["leftover_current"] * waste_kg).round(2)

    df["date"] = df["date"].dt.date
    return df.drop(columns="dow").sort_values(["date", "meal", "dish"]).reset_index(drop=True)


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    crowd, demand = generate()
    crowd.to_csv(config.CROWD_CSV, index=False, encoding="utf-8-sig")
    demand.to_csv(config.DEMAND_CSV, index=False, encoding="utf-8-sig")
    print(f"已產生 {config.CROWD_CSV.name}：{len(crowd):,} 筆")
    print(f"已產生 {config.DEMAND_CSV.name}：{len(demand):,} 筆")
    print(f"目前備餐方式總剩食：{demand['waste_kg_current'].sum():,.0f} kg")


if __name__ == "__main__":
    main()
