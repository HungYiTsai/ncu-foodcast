"""
真實資料匯入：檢查使用者上傳的 CSV 格式、自動補上行事曆特徵，並可備份／還原模擬資料。

這是把專題從「模擬資料」走向「真實學餐資料」的橋樑：
店家只要提供最基本的欄位（日期、餐別、時段或品項、人數或銷量），
其餘的行事曆特徵（星期幾、考試週、國定假日……）由程式自動算出。
"""
from __future__ import annotations

import shutil

import pandas as pd

from foodcast import config
from foodcast.calendar_tw import day_features

BACKUP_DIR = config.DATA_DIR / "simulated_backup"

# 每種資料必填的欄位；天氣與活動欄位可省略，省略時用預設值補上
REQUIRED = {
    "crowd": ["date", "meal", "slot", "customers"],
    "demand": ["date", "meal", "dish", "demand"],
}
OPTIONAL_DEFAULTS = {"temp": None, "rain_mm": 0.0, "has_event": 0, "is_promo": 0}
TARGET = {"crowd": "customers", "demand": "demand"}


def template(kind: str) -> pd.DataFrame:
    """產生範本 CSV 讓店家照格式填寫。"""
    if kind == "crowd":
        return pd.DataFrame({
            "date": ["2026-11-02"] * 3, "meal": ["午餐"] * 3, "slot": ["11:30", "12:00", "12:30"],
            "customers": [210, 430, 300], "temp": [22.5] * 3, "rain_mm": [0.0] * 3, "has_event": [0] * 3,
        })
    return pd.DataFrame({
        "date": ["2026-11-02"] * 2, "meal": ["午餐"] * 2, "dish": ["便當", "麵食"],
        "demand": [240, 180], "prep_current": [260, 200], "is_promo": [0, 1],
        "temp": [22.5] * 2, "rain_mm": [0.0] * 2, "has_event": [0] * 2,
    })


def validate(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame | None, list[str]]:
    """
    檢查上傳的資料。回傳 (整理後的 DataFrame 或 None, 錯誤/提醒訊息清單)。
    有任何「錯誤」時回傳 None；只有「提醒」時仍會回傳資料。
    """
    errors, notes = [], []
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        return None, [f"❌ 缺少必要欄位：{', '.join(missing)}"]

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        errors.append(f"❌ 有 {int(out['date'].isna().sum())} 列日期格式錯誤（請用 YYYY-MM-DD）")

    bad_meal = sorted(set(out["meal"]) - set(config.MEALS))
    if bad_meal:
        errors.append(f"❌ meal 只能是「午餐」或「晚餐」，發現：{bad_meal}")

    if kind == "crowd":
        valid_slots = {s for m in config.MEALS for s in config.SLOTS[m]}
        bad = sorted(set(out["slot"].astype(str)) - valid_slots)
        if bad:
            errors.append(f"❌ slot 必須是 {sorted(valid_slots)} 之一，發現：{bad}")
    else:
        bad = sorted(set(out["dish"]) - set(config.DISH_NAMES))
        if bad:
            errors.append(f"❌ dish 必須是 {config.DISH_NAMES} 之一，發現：{bad}"
                          "（新品項請先加到 foodcast/config.py 的 DISHES）")

    y = pd.to_numeric(out[TARGET[kind]], errors="coerce")
    if y.isna().any() or (y < 0).any():
        errors.append(f"❌ {TARGET[kind]} 必須是 0 以上的數字")
    if errors:
        return None, errors
    out[TARGET[kind]] = y.round().astype(int)

    # 補上可省略的欄位
    for col, default in OPTIONAL_DEFAULTS.items():
        if col == "is_promo" and kind == "crowd":
            continue
        if col not in out.columns:
            if col == "temp":
                from foodcast.predict import default_weather
                out[col] = out["date"].map(lambda d: default_weather(d.date())["temp"])
            else:
                out[col] = default
            notes.append(f"ℹ️ 沒有 {col} 欄位，已用預設值補上")

    # 自動補上行事曆特徵（星期幾、考試週、假日……）
    cal = pd.DataFrame([day_features(d.date()) for d in out["date"]], index=out.index)
    out = pd.concat([out.drop(columns=[c for c in cal.columns if c in out.columns]), cal], axis=1)

    n_days = out["date"].nunique()
    if n_days < 60:
        notes.append(f"⚠️ 只有 {n_days} 天的資料，建議至少一個學期（約 120 天）模型才會穩定")

    if kind == "demand":
        out["date"] = out["date"].dt.date
        if "prep_current" not in out.columns:
            from foodcast.data_generator import add_current_practice
            out = add_current_practice(out)
            notes.append("ℹ️ 沒有 prep_current（實際備餐量），已用「前兩週同一天 × 1.15」估計")
        else:
            waste_kg = out["dish"].map({k: v["waste_kg"] for k, v in config.DISHES.items()})
            out["leftover_current"] = (out["prep_current"] - out["demand"]).clip(lower=0)
            out["stockout_current"] = (out["demand"] - out["prep_current"]).clip(lower=0)
            out["waste_kg_current"] = (out["leftover_current"] * waste_kg).round(2)
        out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True), notes


def backup_simulated() -> None:
    """第一次替換資料前，先把模擬資料備份起來。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in (config.CROWD_CSV, config.DEMAND_CSV):
        dst = BACKUP_DIR / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def restore_simulated() -> bool:
    """還原成模擬資料。成功回傳 True。"""
    ok = False
    for name in (config.CROWD_CSV.name, config.DEMAND_CSV.name):
        src = BACKUP_DIR / name
        if src.exists():
            shutil.copy2(src, config.DATA_DIR / name)
            ok = True
    return ok


def save(df: pd.DataFrame, kind: str) -> None:
    backup_simulated()
    path = config.CROWD_CSV if kind == "crowd" else config.DEMAND_CSV
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out.to_csv(path, index=False, encoding="utf-8-sig")
