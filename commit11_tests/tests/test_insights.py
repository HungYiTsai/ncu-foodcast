"""
第二批擴充功能測試：預測原因解釋、成效換算、錯誤分析、學生回報。
執行：python -m pytest -q
"""
import datetime as dt
from pathlib import Path

import pytest

from foodcast import config, error_analysis, explain, feedback, impact, predict

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def crowd_model():
    return predict.load_models()[0]


# ---------------- 預測原因解釋 ----------------
def test_normal_day_has_no_factors(crowd_model):
    # 學期中的週三、沒下雨、月平均氣溫 → 沒有特殊因素
    d = dt.date(2026, 11, 18)
    df = explain.explain_crowd(crowd_model, d, predict.default_weather(d)["temp"], 0)
    assert df.empty
    assert "一般日" in explain.explain_sentence(df, "午餐")


def test_rain_increases_and_weekend_decreases(crowd_model):
    d = dt.date(2026, 11, 21)   # 週六
    df = explain.explain_crowd(crowd_model, d, predict.default_weather(d)["temp"], 20)
    rain = df[df["factor"].str.startswith("下雨")]
    weekend = df[df["factor"] == "週六"]
    assert (rain["effect"] > 0).all()
    assert (weekend["effect"] < 0).all()


def test_break_is_listed(crowd_model):
    df = explain.explain_crowd(crowd_model, dt.date(2027, 2, 3), 16, 0)
    assert "寒暑假" in set(df["factor"])


# ---------------- 成效換算 ----------------
def test_impact_summary_consistent():
    bt = impact.load_backtest()
    s = impact.summary(bt)
    assert s["saved_money"] > 0
    assert s["current"]["total_loss"] == pytest.approx(
        s["current"]["waste_cost"] + s["current"]["lost_profit"], rel=1e-6)
    # 剩食公斤數要和 train.py 的回測結果一致
    m = predict.load_metrics()["demand"]["backtest"]
    assert s["current"]["waste_kg"] == pytest.approx(m["current"]["waste_kg"], abs=0.5)
    assert s["ai"]["waste_kg"] == pytest.approx(m["ai"]["waste_kg"], abs=0.5)


def test_higher_service_level_less_stockout():
    mq = impact.money_by_quantile(impact.load_backtest())
    assert mq["缺貨損失"].is_monotonic_decreasing
    assert mq["剩食損失"].is_monotonic_increasing


def test_by_dish_covers_all_dishes():
    assert impact.by_dish(impact.load_backtest())["品項"].tolist() == config.DISH_NAMES


# ---------------- 錯誤分析 ----------------
def test_error_analysis_tables():
    err = error_analysis.load_errors()
    assert err["in_semester"].notna().all()          # 特徵有成功合併
    sc = error_analysis.by_scenario(err)
    assert sc["筆數"].sum() == len(err)                 # 每一列剛好歸到一個情境
    assert len(error_analysis.worst_days(err, n=5)) == 5


# ---------------- 學生回報 ----------------
def test_feedback_roundtrip(tmp_path):
    path = tmp_path / "feedback.csv"     # 用暫存檔，不影響真正的回報資料
    feedback.save(dt.date(2026, 11, 27), "午餐", "12:00", "大排長龍（15 分鐘以上）", "擁擠", 350, path=path)
    feedback.save(dt.date(2026, 11, 27), "晚餐", "18:00", "排一下（5 分鐘內）", "普通", 180, path=path)
    df = feedback.load(path)
    assert len(df) == 2
    ag = feedback.agreement(df)
    assert ag == {"n": 2, "exact": 0.5, "within1": 1.0, "bias": 0.5}


def test_nearest_slot():
    assert feedback.nearest_slot(dt.datetime(2026, 11, 27, 12, 10)) == ("午餐", "12:00")
    assert feedback.nearest_slot(dt.datetime(2026, 11, 27, 18, 40)) == ("晚餐", "18:30")


# ---------------- 新頁面 ----------------
@pytest.mark.parametrize("page", ["pages/impact.py", "pages/student.py", "pages/analysis.py"])
def test_pages_render(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception
