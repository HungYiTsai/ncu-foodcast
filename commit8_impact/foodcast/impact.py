"""
成效換算：把回測結果換算成店家與學校看得懂的數字——省下多少錢、減少多少碳排。

成本假設（皆為估計值，在 config.py 的 DISHES 調整）：
  - 剩食損失 = 剩下的份數 × 每份食材成本（cost）
  - 缺貨損失 = 賣完而沒賣到的份數 × 每份毛利（price - cost）
  - 總損失   = 剩食損失 + 缺貨損失
  備餐太多會浪費食材，備餐太少會少賺錢，兩種損失都要算，才能公平比較。
"""
from __future__ import annotations

import pandas as pd

from foodcast import config


def load_backtest() -> pd.DataFrame:
    return pd.read_csv(config.MODEL_DIR / "demand_backtest.csv", parse_dates=["date"])


def policy_costs(bt: pd.DataFrame, prep_col: str) -> pd.DataFrame:
    """逐列計算某種備餐策略的剩食份數、剩食公斤、剩食損失、缺貨損失。"""
    d = config.DISHES
    out = bt[["date", "meal", "dish", "demand"]].copy()
    out["prep"] = bt[prep_col]
    out["leftover"] = (out["prep"] - out["demand"]).clip(lower=0)
    out["stockout"] = (out["demand"] - out["prep"]).clip(lower=0)
    out["waste_kg"] = out["leftover"] * out["dish"].map({k: v["waste_kg"] for k, v in d.items()})
    out["waste_cost"] = out["leftover"] * out["dish"].map({k: v["cost"] for k, v in d.items()})
    out["lost_profit"] = out["stockout"] * out["dish"].map({k: v["price"] - v["cost"] for k, v in d.items()})
    out["total_loss"] = out["waste_cost"] + out["lost_profit"]
    return out


def summary(bt: pd.DataFrame, quantile: float = config.DEFAULT_QUANTILE) -> dict:
    """比較「目前方式」與「AI 建議」的整體成效。"""
    cur = policy_costs(bt, "prep_current")
    ai = policy_costs(bt, f"prep_q{int(quantile * 100)}")
    cols = ["leftover", "stockout", "waste_kg", "waste_cost", "lost_profit", "total_loss"]
    c, a = cur[cols].sum(), ai[cols].sum()
    days = bt["date"].nunique()
    saved_money = float(c["total_loss"] - a["total_loss"])
    saved_kg = float(c["waste_kg"] - a["waste_kg"])
    return {
        "days": int(days),
        "current": {k: round(float(v), 1) for k, v in c.items()},
        "ai": {k: round(float(v), 1) for k, v in a.items()},
        "saved_money": round(saved_money),
        "saved_money_per_day": round(saved_money / days),
        "saved_kg": round(saved_kg, 1),
        "saved_servings": int(c["leftover"] - a["leftover"]),
        "co2e_kg": round(saved_kg * config.CO2E_PER_KG_FOOD_WASTE, 1),
        # 用測試期間（含週末）的日平均 × 365 天粗估一年；寒暑假人少，實際可能較低
        "yearly_money_estimate": round(saved_money / days * 365),
    }


def monthly(bt: pd.DataFrame, quantile: float = config.DEFAULT_QUANTILE) -> pd.DataFrame:
    """每月的剩食公斤數與總損失，比較兩種方式。"""
    frames = []
    for label, col in [("目前方式", "prep_current"), ("AI 建議", f"prep_q{int(quantile * 100)}")]:
        p = policy_costs(bt, col)
        p["月份"] = p["date"].dt.strftime("%Y-%m")
        g = p.groupby("月份")[["waste_kg", "total_loss"]].sum().reset_index()
        g["方式"] = label
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def by_dish(bt: pd.DataFrame, quantile: float = config.DEFAULT_QUANTILE) -> pd.DataFrame:
    """各品項的剩食減少量與省下的錢。"""
    cur = policy_costs(bt, "prep_current").groupby("dish")[["waste_kg", "total_loss"]].sum()
    ai = policy_costs(bt, f"prep_q{int(quantile * 100)}").groupby("dish")[["waste_kg", "total_loss"]].sum()
    out = pd.DataFrame({
        "目前剩食 (kg)": cur["waste_kg"].round(0),
        "AI 剩食 (kg)": ai["waste_kg"].round(0),
        "剩食減少 (%)": (100 * (1 - ai["waste_kg"] / cur["waste_kg"])).round(1),
        "省下金額 (NT$)": (cur["total_loss"] - ai["total_loss"]).round(0),
    })
    return out.reindex(config.DISH_NAMES).reset_index(names="品項")


def money_by_quantile(bt: pd.DataFrame) -> pd.DataFrame:
    """每種服務水準的總損失（剩食損失＋缺貨損失），找出最省錢的設定。"""
    rows = []
    for q in config.QUANTILES:
        p = policy_costs(bt, f"prep_q{int(q * 100)}")
        rows.append({"服務水準": f"{int(q * 100)}%", "quantile": q,
                     "剩食損失": p["waste_cost"].sum(), "缺貨損失": p["lost_profit"].sum(),
                     "總損失": p["total_loss"].sum()})
    return pd.DataFrame(rows)
