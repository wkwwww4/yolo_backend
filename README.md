# yolo_backend

此專案為簡單的人流（行人）檢測與追蹤範例，使用 YOLOv8 與 ByteTrack 進行檢測與追蹤，會輸出每個影片的「人流統計報告」。

**主要檔案**
- `检测人流.py`：核心腳本，負責批次掃描 `video/` 資料夾中的影片，使用 YOLO 偵測行人並用 ByteTrack 跟蹤，最後產生報表與（可選）標註影片。
- `yolov8n.pt`：YOLOv8 預訓練模型檔（已放在 repo 中）。
- `video/`：請將欲分析的影片放到此資料夾（支援 `.mp4`, `.avi`, `.mov`, `.mkv`, `.hevc` 等）。
- `tracking_results/`：腳本執行時會自動建立，內含輸出的統計報告與標註影片（若有啟用）。

**功能簡述**
- 批次處理 `video/` 中的所有影片
- 使用 YOLO 只偵測行人（類別 0）
- 使用 ByteTrack 分配唯一 ID 並統計出現過的不同人數
- 輸出：每支影片的標註影片（可選）與 `tracking_results/人流统计报告.txt`

**快速開始（建議）**
1. 建議建立虛擬環境並啟用：

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安裝必要套件：

```bash
pip install -U pip
pip install ultralytics opencv-python-headless pillow numpy lap
```

3. 將欲分析的影片放到 `video/`（若無此資料夾請建立）：

```bash
mkdir -p video
# 把影片複製或上傳到 video/ 內
```

4. 執行腳本：

```bash
python 检测人流.py
```

執行後會在 `tracking_results/` 產生：
- `*_人流统计.mp4`（帶標註的輸出影片，若輸出被啟用）
- `人流统计报告.txt`（所有影片的統計摘要）

**若只需要報表而不輸出標註影片**
- 目前腳本預設會寫出標註影片。如果你只需要 `人流统计报告.txt`，有兩種選擇：
	1. 在執行前暫時將 `检测人流.py` 中寫檔（`out.write(frame)`）相關程式碼註解或移除。
	2. 我可以幫你修改腳本，新增命令列參數 `--no-video`（或 `--save-video False`）以在執行時選擇是否輸出影片。

**常見問題 & 排錯**
- 如果出現 `No module named 'lap'`，請執行：

```bash
pip install lap
```

- 如果影片無法開啟，請檢查檔案副檔名與檔案是否損毀，或者使用 `ffmpeg` 轉檔到常見格式再重試。

**範例流程**

```bash
# 1. 建立 venv、安裝套件
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python-headless pillow numpy lap

# 2. 把要檢測的影片放到 video/，再執行
python 检测人流.py

# 3. 查看結果
ls tracking_results
cat tracking_results/人流统计报告.txt
```


