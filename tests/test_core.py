"""
核心功能測試：python -m pytest -q

測試涵蓋：行事曆規則、模擬資料合理性、預測輸出格式、AI 助理意圖判斷，
以及四個 Streamlit 頁面都能正常開啟（不會報錯）。
"""
import datetime as dt
from pathlib import Path

import pytest

from foodcast import assistant, config, predict
from foodcast.calendar_tw import academic_info, day_features

ROOT = Path(__file__).resolve().parent.parent


# ---------------- 行事曆 ----------------
def test_semester_weeks():
    assert academic_info(dt.date(2025, 9, 8))["week"] == 1            # 114 上學期開學
    assert academic_info(dt.date(2025, 11, 3))["exam_week"] == 1      # 第 9 週期中考
    assert academic_info(dt.date(2026, 1, 5))["exam_week"] == 1       # 第 18 週期末考
    assert academic_info(dt.date(2026, 2, 1))["in_semester"] == 0     # 寒假
    assert day_features(dt.date(2025, 10, 10))["is_holiday"] == 1     # 國慶日


# ---------------- 模擬資料 ----------------
def test_simulated_data_sanity():
    from foodcast.data_generator import generate
    crowd, demand = generate(seed=0)
    assert (crowd["customers"] >= 0).all() and (demand["demand"] >= 0).all()
    # 學期中平日午餐應該比寒暑假多
    weekday = crowd[(crowd["weekday"] < 5) & (crowd["meal"] == "午餐")]
    sem = weekday[weekday["in_semester"] == 1].groupby("date")["customers"].sum().mean()
    brk = weekday[weekday["in_semester"] == 0].groupby("date")["customers"].sum().mean()
    assert sem > 2 * brk
    # 12:00 應該是午餐尖峰
    peak = crowd[crowd["meal"] == "午餐"].groupby("slot")["customers"].mean().idxmax()
    assert peak == "12:00"


# ---------------- 預測 ----------------
@pytest.fixture(scope="module")
def models():
    return predict.load_models()


def test_forecast_outputs(models):
    crowd_model, demand_models = models
    d = dt.date(2026, 11, 18)  # 學期中的週三
    c = predict.forecast_crowd(crowd_model, d, 21, 0)
    assert len(c) == 10 and set(c["level"]) <= {"舒適", "普通", "擁擠", "爆滿"}
    dm = predict.forecast_demand(demand_models, d, 21, 0)
    assert len(dm) == 12
    assert (dm["recommend"] >= dm["expected"]).all()


def test_rain_increases_crowd(models):
    crowd_model, _ = models
    d = dt.date(2026, 11, 18)
    dry = predict.forecast_crowd(crowd_model, d, 21, 0)["customers"].sum()
    wet = predict.forecast_crowd(crowd_model, d, 21, 20)["customers"].sum()
    assert wet > dry


def test_backtest_beats_current_practice():
    bt = predict.load_metrics()["demand"]["backtest"]
    assert bt["ai"]["waste_kg"] < bt["current"]["waste_kg"]


# ---------------- AI 助理 ----------------
def test_parse_date():
    today = dt.date(2026, 11, 16)  # 週一
    assert assistant.parse_date("明天中午", today) == dt.date(2026, 11, 17)
    assert assistant.parse_date("週五晚餐", today) == dt.date(2026, 11, 20)
    assert assistant.parse_date("週日", today) == dt.date(2026, 11, 22)
    assert assistant.parse_date("12/3 午餐", today) == dt.date(2026, 12, 3)
    assert assistant.parse_date("人多嗎", today) is None


def test_intent():
    assert assistant.detect_intent("幾點去人最少") == "when"
    assert assistant.detect_intent("便當要備幾份") == "prep"
    assert assistant.detect_intent("推薦吃什麼") == "dish"


def test_offline_answer(models, monkeypatch):
    for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    crowd_model, demand_models = models

    def summary(d):
        return predict.day_summary(d, predict.forecast_crowd(crowd_model, d, 21, 0),
                                   predict.forecast_demand(demand_models, d, 21, 0), 21, 0)

    reply, mode = assistant.answer("明天中午幾點去人最少？", dt.date(2026, 11, 16),
                                   dt.date(2026, 11, 16), summary)
    assert mode == "離線" and "午餐最推薦" in reply


# ---------------- Streamlit 頁面 ----------------
@pytest.mark.parametrize("page", ["app.py", "pages/home.py", "pages/student.py", "pages/store.py",
                                  "pages/analysis.py", "pages/assistant.py"])
def test_pages_render(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception
