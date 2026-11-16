"""
專案共用設定：檔案路徑、餐點品項、時段與成本參數。

所有模組都從這裡讀設定，要調整品項或成本時只需要改這一個檔案。
"""
import os
from pathlib import Path

# ---- 路徑 ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

CROWD_CSV = DATA_DIR / "crowd.csv"      # 每 30 分鐘時段的來客數
DEMAND_CSV = DATA_DIR / "demand.csv"    # 每餐每品項的需求量與備餐量
CROWD_MODEL = MODEL_DIR / "crowd_model.joblib"
DEMAND_MODELS = MODEL_DIR / "demand_models.joblib"
METRICS_JSON = MODEL_DIR / "metrics.json"

# ---- 模擬資料期間（兩個學年度，讓模型看過兩次期中/期末週）----
SIM_START = "2024-09-01"
SIM_END = "2026-06-30"
# 時間序列切分：之前的資料訓練，之後的資料測試（避免用未來資料預測過去）
TEST_START = "2026-03-01"

RANDOM_SEED = 42

# ---- 用餐時段 ----
MEALS = ["午餐", "晚餐"]
SLOTS = {
    "午餐": ["11:00", "11:30", "12:00", "12:30", "13:00"],
    "晚餐": ["17:00", "17:30", "18:00", "18:30", "19:00"],
}

# ---- 品項設定 ----
# share：平日基準點餐比例；waste_kg：每份剩食重量；
# cost：每份食材成本（元）；price：售價（元）
DISHES = {
    "自助餐":   {"share": 0.30, "waste_kg": 0.45, "cost": 45, "price": 85},
    "便當":     {"share": 0.20, "waste_kg": 0.40, "cost": 40, "price": 80},
    "麵食":     {"share": 0.18, "waste_kg": 0.50, "cost": 30, "price": 70},
    "咖哩飯":   {"share": 0.12, "waste_kg": 0.45, "cost": 35, "price": 75},
    "素食":     {"share": 0.08, "waste_kg": 0.40, "cost": 35, "price": 75},
    "輕食沙拉": {"share": 0.12, "waste_kg": 0.25, "cost": 45, "price": 90},
}
DISH_NAMES = list(DISHES.keys())

# ---- 備餐建議的服務水準 ----
# 用分位數迴歸 (quantile regression) 預測「需求的第 q 分位數」作為建議備餐量。
# q 越高越不容易賣完，但剩食越多；店家可在介面上自行選擇。
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_QUANTILE = 0.7

# 每公斤廚餘約等於多少公斤 CO2e（估計值，僅供呈現量級；可依正式文獻替換）
CO2E_PER_KG_FOOD_WASTE = 2.5


def load_env_file(path: Path = ROOT_DIR / ".env") -> None:
    """
    讀取專案根目錄的 .env（若存在），把金鑰放進環境變數。
    用簡單的寫法取代 python-dotenv，少裝一個套件；已存在的環境變數不會被覆蓋。
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
