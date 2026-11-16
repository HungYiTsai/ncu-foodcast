"""
AI 助理：用自然語言回答「明天中午幾點去人最少？」「晚餐便當要備幾份？」這類問題。

兩種模式：
  1. LLM 模式：環境變數有設定 Azure OpenAI 或 OpenAI 金鑰時，
     把模型預測結果（JSON）交給大型語言模型，請它用口語回答。
     LLM 只負責「說明」，數字一律來自我們自己訓練的模型，避免 LLM 自己編數字（幻覺）。
  2. 離線模式（預設）：沒有金鑰時，用關鍵字判斷意圖，套用回答模板。
     確保沒有網路或金鑰時，Demo 一樣能運作。

金鑰設定方式見 .env.example，金鑰絕對不要上傳到 GitHub。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Callable

from foodcast import config

SYSTEM_PROMPT = """你是中央大學學餐的 AI 助理「惜食預報」。
你會收到一份 JSON，內容是我們自己訓練的機器學習模型對某一天的預測（各時段人數、各品項建議備餐量）。
規則：
1. 只能使用 JSON 裡的數字，不可以自己估算或編造數字。
2. 用繁體中文、口語、簡短（3 句以內）回答。
3. 學生問人潮：給出最推薦的時段與原因。店家問備餐：給出建議份數，並提醒這是預測值。
4. JSON 沒有的資訊就說不知道。"""


# ------------------------------------------------------------------ 日期解析
def parse_date(question: str, today: dt.date) -> dt.date | None:
    """從問題中找出日期：今天/明天/後天/週X/M/D，找不到就回傳 None。"""
    if "今天" in question or "今日" in question:
        return today
    if "後天" in question:
        return today + dt.timedelta(days=2)
    if "明天" in question or "明日" in question:
        return today + dt.timedelta(days=1)
    m = re.search(r"(\d{1,2})[/月](\d{1,2})", question)
    if m:
        try:
            return dt.date(today.year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    m = re.search(r"(?:週|星期|禮拜)([一二三四五六日天])", question)
    if m:
        target = min("一二三四五六日天".index(m.group(1)), 6)   # 「日」「天」都是週日
        return today + dt.timedelta(days=(target - today.weekday()) % 7)
    return None


def detect_meal(question: str) -> str | None:
    if any(k in question for k in ["午餐", "中午", "午飯"]):
        return "午餐"
    if any(k in question for k in ["晚餐", "晚上", "晚飯"]):
        return "晚餐"
    return None


def detect_intent(question: str) -> str:
    """簡單的關鍵字意圖判斷。"""
    if any(k in question for k in ["備", "準備", "幾份", "進貨", "剩食", "浪費"]):
        return "prep"
    if any(k in question for k in ["幾點", "什麼時候", "人少", "排隊", "擁擠", "人多", "爆滿", "時段"]):
        return "when"
    if any(k in question for k in ["吃什麼", "推薦", "熱門", "賣最好"]):
        return "dish"
    return "summary"


# ------------------------------------------------------------------ 離線模板回答
def rule_based_answer(question: str, s: dict) -> str:
    intent = detect_intent(question)
    meal = detect_meal(question)
    meals = [meal] if meal else config.MEALS
    head = f"{s['date']}（週{s['weekday']}）"
    if s["tags"]:
        head += "，" + "、".join(s["tags"])
    if s["weather"]["rain_mm"] > 0:
        head += "，預報有雨"

    lines = [head + "："]
    for m in meals:
        info = s["meals"][m]
        if intent == "when":
            lines.append(f"・{m}最推薦 {info['quiet_slot']} 去（{info['quiet_level']}），"
                         f"避開 {info['peak_slot']} 的尖峰（{info['peak_level']}）。")
        elif intent == "prep":
            dishes = "、".join(f"{d} {v['recommend']} 份" for d, v in info["dishes"].items())
            lines.append(f"・{m}建議備餐：{dishes}。")
        elif intent == "dish":
            top = max(info["dishes"].items(), key=lambda kv: kv[1]["expected"])
            lines.append(f"・{m}預計最熱門的是{top[0]}（約 {top[1]['expected']} 份），想吃要早點去。")
        else:
            lines.append(f"・{m}預計約 {info['total_customers']} 人，尖峰 {info['peak_slot']}"
                         f"（{info['peak_level']}），建議 {info['quiet_slot']} 前往。")
    if intent == "prep":
        lines.append("（建議量為模型預測值，可依當天實際狀況微調。）")
    return "\n".join(lines)


# ------------------------------------------------------------------ LLM 模式
def llm_backend() -> str | None:
    """偵測可用的 LLM 服務；都沒有就回傳 None（使用離線模式）。"""
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def llm_answer(question: str, s: dict) -> str:
    from openai import AzureOpenAI, OpenAI  # 選用套件，只有 LLM 模式才需要安裝

    backend = llm_backend()
    if backend == "azure":
        client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                             api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
        model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"預測資料：{json.dumps(s, ensure_ascii=False)}\n\n問題：{question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, today: dt.date, default_day: dt.date,
           get_summary: Callable[[dt.date], dict]) -> tuple[str, str]:
    """
    回答問題，回傳 (回答內容, 使用的模式)。
    today：真實的今天，用來解析「明天」「週五」；問題沒提到日期時用 default_day（側邊欄選的日期）。
    get_summary(date) 由介面提供，負責產生該日期的預測摘要。
    LLM 呼叫失敗（網路、額度等問題）時自動退回離線模式，Demo 不會中斷。
    """
    d = parse_date(question, today) or default_day
    s = get_summary(d)
    if llm_backend():
        try:
            return llm_answer(question, s), "LLM"
        except Exception as e:  # noqa: BLE001 — 任何錯誤都退回離線模式
            return rule_based_answer(question, s) + f"\n\n（LLM 暫時無法使用，已改用離線模式：{type(e).__name__}）", "離線"
    return rule_based_answer(question, s), "離線"
