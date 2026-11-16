"""
模型訓練與評估。

1. 人潮模型：預測每 30 分鐘時段的來客數（迴歸問題）
   - 比較 4 種方法：歷史平均（基準線）、線性迴歸、隨機森林、梯度提升樹
   - 以時間切分的測試集 MAE 選出最佳模型
2. 備餐模型：預測每餐每品項需求量的「分位數」(quantile regression)
   - 例如 q=0.7 代表「有 70% 的機率需求不會超過這個數字」
   - 用分位數當建議備餐量，可以在剩食與缺貨之間取捨
3. 回測 (backtest)：在測試期間比較「目前備餐方式」與「AI 建議」的剩食與缺貨

執行：python -m foodcast.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from foodcast import config, data_generator
from foodcast.features import (CROWD_FEATURES, DEMAND_CATEGORICAL, DEMAND_FEATURES,
                               FEATURE_LABELS, crowd_X, demand_X)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀取資料；若資料不存在就先產生模擬資料。"""
    if not (config.CROWD_CSV.exists() and config.DEMAND_CSV.exists()):
        data_generator.main()
    crowd = pd.read_csv(config.CROWD_CSV, parse_dates=["date"])
    demand = pd.read_csv(config.DEMAND_CSV, parse_dates=["date"])
    return crowd, demand


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依日期切分訓練/測試集。時間序列資料不能隨機切分，否則會用未來預測過去。"""
    cut = pd.Timestamp(config.TEST_START)
    return df[df["date"] < cut].copy(), df[df["date"] >= cut].copy()


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


# ------------------------------------------------------------------ 人潮模型
def train_crowd(crowd: pd.DataFrame) -> tuple[object, dict, pd.DataFrame]:
    train, test = time_split(crowd)
    X_tr, y_tr = crowd_X(train), train["customers"]
    X_te, y_te = crowd_X(test), test["customers"]

    results = {}

    # (a) 基準線：訓練期間「同星期幾、同時段」的平均來客數
    #     任何 ML 模型都要贏過這個簡單方法才有意義
    key = ["weekday", "meal", "slot"]
    avg = train.groupby(key)["customers"].mean().rename("pred")
    base_pred = test.join(avg, on=key)["pred"].fillna(y_tr.mean())
    results["歷史平均（基準線）"] = regression_metrics(y_te, base_pred)

    candidates = {
        "線性迴歸": LinearRegression(),
        "隨機森林": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      random_state=config.RANDOM_SEED, n_jobs=-1),
        "梯度提升樹": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                               random_state=config.RANDOM_SEED),
    }
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        results[name] = regression_metrics(y_te, model.predict(X_te))
        fitted[name] = model

    best_name = min(fitted, key=lambda n: results[n]["MAE"])
    best = fitted[best_name]

    # 置換重要性 (permutation importance)：打亂某個特徵後誤差變大越多，代表越重要
    imp = permutation_importance(best, X_te, y_te, n_repeats=5,
                                 random_state=config.RANDOM_SEED, scoring="neg_mean_absolute_error")
    importance = sorted(
        ({"feature": FEATURE_LABELS[f], "importance": round(float(v), 2)}
         for f, v in zip(CROWD_FEATURES, imp.importances_mean)),
        key=lambda r: -r["importance"],
    )

    # 用全部資料重新訓練最終模型（評估已完成，部署時用越多資料越好）
    final = type(best)(**best.get_params()).fit(crowd_X(crowd), crowd["customers"])

    test_out = test[["date", "meal", "slot", "customers"]].copy()
    test_out["pred"] = best.predict(X_te).round()
    info = {"comparison": results, "best_model": best_name, "importance": importance}
    return final, info, test_out


# ------------------------------------------------------------------ 備餐模型
def _quantile_model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
        categorical_features=DEMAND_CATEGORICAL, random_state=config.RANDOM_SEED,
    )


def summarize_policy(df: pd.DataFrame, prep_col: str) -> dict:
    """計算某種備餐策略在測試期間的總剩食、缺貨與滿足率。"""
    waste_kg = df["dish"].map({k: v["waste_kg"] for k, v in config.DISHES.items()})
    leftover = (df[prep_col] - df["demand"]).clip(lower=0)
    stockout = (df["demand"] - df[prep_col]).clip(lower=0)
    return {
        "prepared": int(df[prep_col].sum()),
        "leftover_servings": int(leftover.sum()),
        "waste_kg": round(float((leftover * waste_kg).sum()), 1),
        "stockout_servings": int(stockout.sum()),
        # 滿足率：有多少比例的點餐需求被滿足（沒有遇到賣完）
        "fill_rate": round(float(1 - stockout.sum() / df["demand"].sum()), 4),
    }


def train_demand(demand: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    train, test = time_split(demand)
    X_tr, y_tr = demand_X(train), train["demand"]
    X_te = demand_X(test)

    test_out = test[["date", "meal", "dish", "demand", "prep_current"]].copy()
    tradeoff = []
    for q in config.QUANTILES:
        m = _quantile_model(q).fit(X_tr, y_tr)
        col = f"prep_q{int(q * 100)}"
        test_out[col] = np.ceil(m.predict(X_te).clip(min=0)).astype(int)
        s = summarize_policy(test_out, col)
        tradeoff.append({"quantile": q, **s})

    # 點預測（q=0.5，中位數）的準確度，給「模型準不準」的問題一個直接答案
    point = regression_metrics(test_out["demand"], test_out["prep_q50"])

    current = summarize_policy(test_out, "prep_current")
    ai = next(t for t in tradeoff if t["quantile"] == config.DEFAULT_QUANTILE)
    days = test_out["date"].nunique()
    backtest = {
        "test_days": int(days),
        "current": current,
        "ai": ai,
        "waste_reduction_pct": round(100 * (1 - ai["waste_kg"] / current["waste_kg"]), 1),
        "co2e_saved_kg": round((current["waste_kg"] - ai["waste_kg"]) * config.CO2E_PER_KG_FOOD_WASTE, 1),
    }

    # 最終模型用全部資料重新訓練
    X_all, y_all = demand_X(demand), demand["demand"]
    final = {q: _quantile_model(q).fit(X_all, y_all) for q in config.QUANTILES}
    info = {"point_accuracy": point, "tradeoff": tradeoff, "backtest": backtest}
    return final, info, test_out


def main() -> None:
    config.MODEL_DIR.mkdir(exist_ok=True)
    crowd, demand = load_data()

    print("訓練人潮模型 ...")
    crowd_model, crowd_info, crowd_test = train_crowd(crowd)
    print("訓練備餐（分位數）模型 ...")
    demand_models, demand_info, demand_test = train_demand(demand)

    joblib.dump(crowd_model, config.CROWD_MODEL)
    joblib.dump(demand_models, config.DEMAND_MODELS)
    crowd_test.to_csv(config.MODEL_DIR / "crowd_test_pred.csv", index=False, encoding="utf-8-sig")
    demand_test.to_csv(config.MODEL_DIR / "demand_backtest.csv", index=False, encoding="utf-8-sig")

    metrics = {"crowd": crowd_info, "demand": demand_info,
               "train_period": f"{config.SIM_START} ~ {config.TEST_START} 前",
               "test_period": f"{config.TEST_START} ~ {config.SIM_END}"}
    config.METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 印出結果摘要 ----
    print("\n[人潮模型比較]（測試集）")
    for name, m in crowd_info["comparison"].items():
        print(f"  {name:<12} MAE={m['MAE']:>7}  RMSE={m['RMSE']:>7}  R2={m['R2']}")
    print(f"  → 最佳模型：{crowd_info['best_model']}")
    bt = demand_info["backtest"]
    print(f"\n[備餐回測]（{bt['test_days']} 天，AI 服務水準 q={config.DEFAULT_QUANTILE}）")
    print(f"  目前方式：剩食 {bt['current']['waste_kg']} kg，滿足率 {bt['current']['fill_rate']:.1%}")
    print(f"  AI 建議 ：剩食 {bt['ai']['waste_kg']} kg，滿足率 {bt['ai']['fill_rate']:.1%}")
    print(f"  剩食減少 {bt['waste_reduction_pct']}%，約減少 {bt['co2e_saved_kg']} kg CO2e")


if __name__ == "__main__":
    main()
