# CLAUDE.md — 專案交接說明（給 Claude Code 讀）

> 這份檔案讓 Claude Code 在新的對話中快速了解專案背景。Claude Code 會自動讀取專案根目錄的 CLAUDE.md。

## 溝通方式
- 一律用**繁體中文（台灣用語）**回覆。
- 使用者是大學生，有用 VS Code 寫 Python 的經驗，但 Git／GitHub 流程可能不熟，指令請一步一步給、說明會出現什麼結果。
- 使用者希望你當「指導夥伴」，一步一步帶他產出接近滿分的期末專題。

## 專題是什麼
- 課程：**機器學習概論** 期末專題（分組），同時參加 **Microsoft × NCU AI Innovation Demo Day**（主題 AI for Better Campus & Life）。
- 題目：**惜食預報 FoodCast** — 預測學餐每 30 分鐘人潮與各品項需求，給學生推薦用餐時段、給店家建議備餐量以減少剩食。
- 技術：Python、pandas、scikit-learn（HistGradientBoosting、RandomForest、LinearRegression、分位數迴歸）、Streamlit、Plotly。LLM（Azure OpenAI / OpenAI）為**選用**，使用者**沒有 API Key**，目前用離線模式。
- 資料是**模擬資料**（`foodcast/data_generator.py`，假設都寫在註解裡）。所有簡報／報告都必須清楚標示這點，不可把回測成效說成真實學餐成效。

## 評分項目與繳交物
| 項目 | 比重 | 繳交 |
|---|---|---|
| 程式功能完整性與創意 | 40% | GitHub repo（含註解、README：簡介、功能、技術、安裝執行、操作流程） |
| 專題提案簡報 | 15% | 微軟競賽最終版簡報 PDF（問題/需求 → AI 應用 → 解決方案 → 成果或影響） |
| AI 協作歷程 | 15% | PDF：工具與用途、代表性 Prompt 與回覆、採納/修改/未採納原因、驗證與修正、學習反思；挑 2–3 個實際案例 |
| GitHub 協作歷程 | 10% | 每位組員都有可辨識的 commit、訊息清楚、看得出版本演進 |
| 期末發表 | 20% | 口頭 10–12 分鐘＋Q&A 3–5 分鐘；含背景需求、AI 與解決方案、Demo、成果特色、分工與挑戰、結論與展望 |

- 發表日期：第 1–6 組 12/4、第 7–12 組 12/11（組別尚未確認）。
- 微軟競賽：12/4 23:59 截止，需繳交 **PowerPoint** 提案簡報。
- **不可把 API Key、密碼上傳 GitHub**（`.env` 已在 .gitignore）。

## 目前進度（2026/11/16）
- ✅ 程式 repo 完成，使用者已在自己的 Windows 電腦成功執行網頁與訓練。
- ✅ README 已更新為使用者電腦實際跑出的數字。
- ⬜ 上傳 GitHub（尚未建立 repo；需要組員名單、GitHub 帳號、確認誰是組長）
- ⬜ 微軟競賽 PowerPoint 提案簡報
- ⬜ AI 協作歷程報告 PDF（素材在 `docs/AI協作紀錄素材.md`）
- ⬜ 期末發表簡報＋Demo 彩排＋Q&A 準備

## 以使用者電腦為準的成果數字（簡報、報告一律用這組）
- 人潮模型（測試集）：歷史平均 MAE 43.04／R² 0.737；線性迴歸 64.07／0.418；隨機森林 17.94／0.951；**梯度提升樹 16.9／0.956（選用）**
- 備餐回測（122 天，q=0.7）：目前方式剩食 16,261 kg、滿足率 94.8% → AI 剩食 5,922 kg、滿足率 97.0%，**剩食減少 63.6%**，估計減碳約 25.8 噸 CO₂e（2.5 kg CO₂e/kg 估計值）
- 分位數取捨：q50 3,494 kg/94.8%；q60 4,490/95.8%；q70 5,922/97.0%；q80 7,945/98.1%；q90 11,232/99.0%

## 使用者電腦環境（Windows + PowerShell）
- `python`／`pip`／`py` 指令在這台電腦**無法使用**（只有 WindowsApps 的假 python）。
- 真正的 Python 是 uv 安裝的 3.14.5：`~\.local\bin\python3.14.exe`。
- 專案已建立虛擬環境 `.venv`，**所有指令請用 `.venv\Scripts\python.exe` 開頭**：
  - 啟動網頁：`.venv\Scripts\python.exe -m streamlit run app.py`
  - 訓練：`.venv\Scripts\python.exe -m foodcast.train`
  - 測試：`.venv\Scripts\python.exe -m pytest -q`

## 重要原則
- **GitHub 協作**：不要由一個人一次上傳全部，也**不要用別人的身分做 commit**。照 `docs/GitHub協作與分工指南.md` 的順序，由各組員用自己的帳號分批提交（branch → PR → merge）。
- **誠實呈現**：模擬資料、估計值都要標示清楚；AI 協作報告只寫真實發生過的案例。
- 修改程式後要跑測試確認沒壞；改了模型或資料要重新訓練，並同步更新 README／簡報數字。
- 專案結構與各檔案用途見 README 第 9 節。
