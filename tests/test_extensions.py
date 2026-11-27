"""
擴充功能測試：超參數調整、真實資料上傳檢查、7 天備餐計畫。
執行：python -m pytest -q
"""
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from foodcast import config, data_io, predict
from foodcast.train import load_data, time_split
from foodcast.tuning import date_folds

ROOT = Path(__file__).resolve().parent.parent


# ---------------- 時間序列交叉驗證 ----------------
def test_date_folds_no_future_leak():
    """每個 fold 的驗證日期都必須晚於所有訓練日期，且同一天不會被拆到兩邊。"""
    crowd, _ = load_data()
    train, _ = time_split(crowd)
    train = train.sort_values(["date", "meal", "slot"]).reset_index(drop=True)
    for tr, va in date_folds(train, n_splits=3):
        tr_dates, va_dates = set(train.loc[tr, "date"]), set(train.loc[va, "date"])
        assert max(tr_dates) < min(va_dates)
        assert not tr_dates & va_dates


def test_tuning_quick_run():
    from foodcast.tuning import run
    out = run(quick=True, save=False)
    assert out["n_combinations"] == 2
    assert set(out["test_mae"]) == {"原本設定", "調整後設定"}
    assert out["results"][0]["cv_mae"] <= out["results"][-1]["cv_mae"]   # 依 CV MAE 排序


# ---------------- 資料上傳檢查 ----------------
def test_template_passes_validation():
    for kind in ("crowd", "demand"):
        df, msgs = data_io.validate(data_io.template(kind), kind)
        assert df is not None, msgs
        assert {"weekday", "exam_week", "in_semester"} <= set(df.columns)   # 自動補上行事曆特徵


def test_validation_catches_errors():
    bad = pd.DataFrame({"date": ["2026-13-01"], "meal": ["早餐"], "dish": ["披薩"], "demand": [-1]})
    df, msgs = data_io.validate(bad, "demand")
    assert df is None
    assert len([m for m in msgs if m.startswith("❌")]) == 4


def test_missing_column():
    df, msgs = data_io.validate(pd.DataFrame({"date": ["2026-11-02"]}), "crowd")
    assert df is None and "缺少必要欄位" in msgs[0]


def test_demand_without_prep_is_estimated():
    raw = pd.read_csv(config.DEMAND_CSV)[["date", "meal", "dish", "demand"]]
    df, msgs = data_io.validate(raw, "demand")
    assert df is not None and "prep_current" in df.columns
    assert any("prep_current" in m for m in msgs)


# ---------------- 7 天備餐計畫 ----------------
def test_weekly_plan_shape():
    _, demand_models = predict.load_models()
    plan = predict.weekly_plan(demand_models, dt.date(2026, 11, 30))
    assert len(plan) == 7 * len(config.MEALS) * len(config.DISH_NAMES)
    assert plan["日期"].nunique() == 7
    assert (plan["recommend"] >= plan["expected"]).all()


def test_weekend_plan_smaller_than_weekday():
    _, demand_models = predict.load_models()
    plan = predict.weekly_plan(demand_models, dt.date(2026, 11, 30))   # 週一開始
    totals = plan.groupby("日期", sort=False)["recommend"].sum().to_list()
    assert min(totals[:5]) > max(totals[5:])                             # 平日 > 週末


# ---------------- 新頁面 ----------------
def test_upload_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages/upload.py"), default_timeout=120).run()
    assert not at.exception, at.exception
