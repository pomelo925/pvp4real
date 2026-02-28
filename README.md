
<h1 align="center">PVP4Real：人類介入下的資料高效率學習</h1>

<p align="center">
  <img src="assets/PVP4Real_Teaser.png" alt="PVP4Real" width="100%">
</p>

本專案包含模擬人類介入的訓練與評估腳本，對應 ICRA 2025 論文與網頁：
<a href="https://metadriverse.github.io/pvp4real/"><b>Webpage</b></a> |
<a href="https://github.com/metadriverse/pvp4real"><b>Code</b></a> |
<a href="https://arxiv.org/pdf/2503.04969"><b>PDF</b></a>

## 專案大致架構

- [pvp4real/pvp](pvp4real/pvp) ：主要訓練與算法實作
  - [pvp4real/pvp/experiments](pvp4real/pvp/experiments) ：各種實驗入口（含 MetaDrive）
  - [pvp4real/pvp/sb3](pvp4real/pvp/sb3) ：內建的 RL/控制演算法
  - [pvp4real/pvp/utils](pvp4real/pvp/utils) ：工具與輔助模組
- [pvp4real/scripts](pvp4real/scripts) ：一鍵啟動實驗的 shell 腳本
- [docker](docker) ：容器化環境（CPU 版本）
- [assets](assets) ：專案圖片與資源

## Usage

### Scripts 腳本說明

[pvp4real/scripts](pvp4real/scripts) 目錄包含以下腳本：

#### 配置文件
- **config.yaml** - 統一配置檔，包含所有腳本的參數設定（控制頻率、速度上限、ROS2 話題、訓練參數等）

#### 工具腳本
- **prepare_checkpoint.py** - 準備檢查點目錄，用於從先前訓練恢復（解壓 .zip 檔並驗證檢查點文件）

#### Stretch3 容器腳本（機器人端）
- **stretch3.standby.sh** - 啟動 Stretch3 基礎驅動與相機節點（必須最先執行）
- **stretch3.hitl.py** - HITL 模式的權限仲裁節點（依據 `/stretch/is_teleop` 決定使用人類或策略指令）
- **stretch3.deploy.py** - 部署模式的權限仲裁節點（僅推理，不訓練）

#### PVP4Real 容器腳本（學習端）
- **pvp.hitl.py** - HITL 在線訓練循環（發布策略指令至 `/pvp/novice_cmd_vel` 並進行學習）
- **pvp.deploy.py** - 部署推理循環（僅載入已訓練策略並發布指令，不進行學習）

### 啟動順序

#### HITL 訓練模式
```bash
# 1. 在 Stretch3 容器中啟動機器人基礎設施
bash pvp4real/scripts/stretch3.standby.sh

# 2. 在 Stretch3 容器中啟動權限仲裁節點
python pvp4real/scripts/stretch3.hitl.py

# 3. 在 PVP4Real 容器中啟動訓練循環
python pvp4real/scripts/pvp.hitl.py
```

#### 部署推理模式
```bash
# 1. 在 Stretch3 容器中啟動機器人基礎設施
bash pvp4real/scripts/stretch3.standby.sh

# 2. 在 Stretch3 容器中啟動部署權限節點
python pvp4real/scripts/stretch3.deploy.py

# 3. 在 PVP4Real 容器中啟動推理循環
python pvp4real/scripts/pvp.deploy.py --checkpoint <path_to_checkpoint.zip>
```
