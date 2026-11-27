# notebooks：資料分析與建模過程

這個資料夾用 Jupyter Notebook 一步一步呈現專題的機器學習流程，每個步驟都有圖表和文字說明。
網頁系統（`app.py`）是「成品」，這裡是「怎麼做出來的過程」。

| 檔案 | 內容 |
|---|---|
| `01_eda.ipynb` | **資料探索（EDA）**：資料品質檢查、時段／星期／學期規律、考試週與下雨的影響、氣溫與品項偏好、特徵相關性，最後整理成對建模的啟示 |
| `02_modeling.ipynb` | **建模與評估**：依時間切分、4 種模型比較、預測 vs 實際、殘差、置換重要性、分位數迴歸與校準檢查、回測 |

## 怎麼看

- **直接在 GitHub 上點開**：圖表和結果都已經存在檔案裡，不用安裝任何東西就能看。
- **自己重新執行**：
  ```bash
  pip install -r requirements.txt     # 已包含 matplotlib 與 jupyter
  python -m foodcast.train            # 先產生資料與模型（只需一次）
  jupyter notebook notebooks/         # 或用 VS Code 直接開啟 .ipynb
  ```
  依序執行每個儲存格（Shift + Enter）。

> ⚠️ 使用模擬資料；不同電腦的套件版本不同，數字可能和檔案中儲存的結果有些微差異。
