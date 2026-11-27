"""
人潮模型的超參數調整：時間序列交叉驗證 (Time Series Cross-Validation) + 網格搜尋。

為什麼不用一般的 K-fold？
  一般 K-fold 會隨機打散資料，等於拿「未來」的日子訓練、預測「過去」，
  分數會太樂觀。時間序列交叉驗證永遠只用「較早的日期」訓練、「較晚的日期」驗證：

    fold 1: [訓練        ][驗證]
    fold 2: [訓練              ][驗證]
    fold 3: [訓練                    ][驗證]

  另外，我們以「日期」為單位切分，同一天的 10 個時段一定在同一邊，
  避免同一天的上午在訓練集、下午在驗證集這種資訊洩漏。

只使用訓練期間的資料做交叉驗證，測試集完全不碰，最後才用測試集確認調整後是否真的變好。

執行：python -m foodcast.tuning          （完整搜尋，約 1～3 分鐘）
      python -m foodcast.tuning --quick  （少量組合，測試用）
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from foodcast import config
from foodcast.features import crowd_X
from foodcast.train import load_data, time_split

TUNING_JSON = config.MODEL_DIR / "tuning.json"

# 要搜尋的超參數組合
PARAM_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
}
QUICK_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31], "min_samples_leaf": [20]}
# train.py 目前使用的設定（sklearn 預設 max_leaf_nodes=31、min_samples_leaf=20）
BASELINE_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20}


def date_folds(df: pd.DataFrame, n_splits: int = 5):
    """以日期為單位的時間序列切分，產生 (訓練列索引, 驗證列索引)。"""
    dates = np.sort(df["date"].unique())
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(dates):
        tr_mask = df["date"].isin(dates[tr_d]).to_numpy()
        va_mask = df["date"].isin(dates[va_d]).to_numpy()
        yield np.where(tr_mask)[0], np.where(va_mask)[0]


def make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, random_state=config.RANDOM_SEED, **params)


def cv_score(df: pd.DataFrame, params: dict, n_splits: int = 5) -> tuple[float, float]:
    """回傳某組超參數在各 fold 的 MAE 平均與標準差。"""
    X, y = crowd_X(df), df["customers"].to_numpy()
    maes = []
    for tr, va in date_folds(df, n_splits):
        m = make_model(params).fit(X.iloc[tr], y[tr])
        maes.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
    return float(np.mean(maes)), float(np.std(maes))


def run(quick: bool = False, save: bool = True) -> dict:
    """執行搜尋；save=False 時不寫檔（測試用，避免覆蓋完整搜尋的結果）。"""
    crowd, _ = load_data()
    train, test = time_split(crowd)
    train = train.sort_values(["date", "meal", "slot"]).reset_index(drop=True)
    grid = QUICK_GRID if quick else PARAM_GRID
    n_splits = 3 if quick else 5

    keys = list(grid)
    results = []
    t0 = time.time()
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        mean, std = cv_score(train, params, n_splits)
        results.append({**params, "cv_mae": round(mean, 2), "cv_std": round(std, 2)})
        print(f"  {params} → CV MAE {mean:.2f} ± {std:.2f}")
    results.sort(key=lambda r: r["cv_mae"])
    best = {k: results[0][k] for k in keys}

    # 最後才用測試集比較「原本設定」與「調整後設定」
    X_tr, y_tr = crowd_X(train), train["customers"]
    X_te, y_te = crowd_X(test), test["customers"]
    test_mae = {
        name: round(float(mean_absolute_error(y_te, make_model(p).fit(X_tr, y_tr).predict(X_te))), 2)
        for name, p in [("原本設定", BASELINE_PARAMS), ("調整後設定", best)]
    }

    out = {
        "n_splits": n_splits,
        "n_combinations": len(results),
        "seconds": round(time.time() - t0, 1),
        "best_params": best,
        "baseline_params": BASELINE_PARAMS,
        "test_mae": test_mae,
        "results": results,
    }
    if save:
        config.MODEL_DIR.mkdir(exist_ok=True)
        TUNING_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="人潮模型超參數調整")
    parser.add_argument("--quick", action="store_true", help="只試少量組合（測試用）")
    args = parser.parse_args()

    print("時間序列交叉驗證 + 網格搜尋 ...")
    out = run(quick=args.quick)
    print(f"\n共 {out['n_combinations']} 組 × {out['n_splits']} folds，耗時 {out['seconds']} 秒")
    print(f"最佳超參數：{out['best_params']}")
    print(f"測試集 MAE：原本 {out['test_mae']['原本設定']} → 調整後 {out['test_mae']['調整後設定']}")
    print(f"結果已存到 {TUNING_JSON.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
