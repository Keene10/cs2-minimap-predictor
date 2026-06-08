# CS-Minimap-Tactical-Dataset macOS 环境配置指南

## 1. 系统要求

- **macOS**: 12.0 (Monterey) 或更高版本
- **Python**: 3.11 或更高版本（本项目需要 Python ≥ 3.11）
- **架构**: Apple Silicon (M1/M2/M3) 或 Intel 均可

## 2. Homebrew 安装基础工具

如果你还没有安装 Homebrew，先执行：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装必要工具：

```bash
brew install python@3.11 git curl
```

## 3. Python 虚拟环境

推荐使用 `venv` 创建隔离环境：

```bash
# 进入项目目录
cd /path/to/CS-Minimap-Tactical-Dataset

# 创建虚拟环境
python3 -m venv venv

# 激活（每次新终端都需要执行）
source venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

## 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 5. 下载 awpy 地图数据

awpy 需要额外的地图数据文件（坐标、雷达图等）：

```bash
awpy get maps
```

如果命令失败（网络问题），确保你的网络可以访问 `https://awpycs.com`，或尝试开启加速器后重试。

数据将下载到 `~/.awpy/maps/` 目录。

## 6. 下载 CS2 Demo 文件

从以下网站下载职业比赛 Demo（`.dem` 格式或压缩包）：

- **HLTV**: https://www.hltv.org/ → 比赛页面 → "GOTV Demo"
- **ESL**: https://pro.eslgaming.com/
- **Faceit**: https://www.faceit.com/

将下载的 `.dem` 文件或 `.rar/.zip` 压缩包放入项目目录：

```
data/demos/
├── match1-m1-dust2.dem
├── match1-m2-mirage.dem
└── ...
```

如果是压缩包，先解压：

```bash
# .rar
bsdtar -xf your-file.rar -C data/demos/

# .zip
unzip your-file.zip -d data/demos/
```

## 7. 快速验证

运行坐标转换测试：

```bash
python3 tests/test_coordinate_transform.py -v
```

解析一个 Demo 测试：

```bash
python3 src/parse_demo.py \
  --input data/demos/your-match.dem \
  --output data/parsed_csv \
  --parse-rate 128
```

渲染一帧测试：

```bash
python3 src/render_minimap.py \
  --csv data/parsed_csv/your-match_round_01.csv \
  --map de_dust2 \
  --timestamps "0,30,60" \
  --output data/frames/test
```

## 8. 常见 macOS 问题排查

### 8.1 Apple Silicon (M1/M2/M3) 的 OpenCV 问题

如果在 `import cv2` 时遇到架构错误：

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

或从 conda 安装：

```bash
conda install -c conda-forge opencv
```

### 8.2 Python 版本过低

如果系统 Python 是 3.9 或更低：

```bash
brew install python@3.11
# 然后在虚拟环境中明确使用 3.11
/opt/homebrew/bin/python3.11 -m venv venv
```

### 8.3 awpy 下载地图数据失败 (ConnectionReset)

awpy 需要从 `https://awpycs.com` 下载地图数据。如果遇到连接重置：

1. 检查网络连接，必要时开启加速器
2. 手动下载后解压到 `~/.awpy/maps/`
3. 或联系项目维护者获取离线数据包

### 8.4 解压 .rar 文件失败

macOS 默认没有 `unrar`，使用 `bsdtar`（已随系统自带）：

```bash
bsdtar -xf file.rar
```

### 8.5 权限问题

如果 pip 安装时出现 Permission Denied：

- 确保已激活虚拟环境（`source venv/bin/activate`）
- 不要使用 `sudo pip install`

## 9. IDE 配置推荐

### VS Code

创建 `.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Parse Demo",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/parse_demo.py",
      "args": [
        "--input", "data/demos/match.dem",
        "--output", "data/parsed_csv",
        "--parse-rate", "128"
      ],
      "console": "integratedTerminal"
    },
    {
      "name": "Generate Dataset",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/generate_dataset.py",
      "args": [
        "--csv-dir", "data/parsed_csv",
        "--summary", "data/parsed_csv/match_summary.json",
        "--output", "data/dataset",
        "--fps", "5"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

### PyCharm

1. 打开项目根目录
2. `Preferences` → `Project` → `Python Interpreter` → 选择虚拟环境的 Python
3. 右键 `src/parse_demo.py` → `Run` → 在弹出的配置中填入命令行参数

## 10. 项目结构速览

```
.
├── src/
│   ├── parse_demo.py          # Demo 解析主脚本
│   ├── coordinate_transform.py # 坐标转换模块
│   ├── render_minimap.py      # 小地图渲染引擎
│   └── generate_dataset.py    # 批量数据集生成
├── tests/
│   └── test_coordinate_transform.py
├── data/
│   ├── demos/                 # 原始 .dem 文件
│   ├── maps/                  # 地图雷达图
│   ├── parsed_csv/            # 解析后的回合 CSV
│   ├── frames/                # 渲染的关键帧图片
│   └── dataset/               # 最终数据集（train/val）
├── map_overview.json          # 地图坐标配置
├── bomb_sites.json            # 炸弹点区域配置
├── requirements.txt
└── SETUP_MACOS.md             # 本文件
```
