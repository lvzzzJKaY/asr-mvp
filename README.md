# Personalized ASR MVP

一个面向构音障碍场景的个性化语音识别 MVP。

## 功能

- 训练短语（个性化热词）
- 上传音频识别
- 浏览器录音识别
- 候选确认并回写
- 短语板管理
- 本地 profile 导出/删除

## 运行环境

- Python 3.13（建议）
- macOS / Linux / Windows
- 已安装依赖（见 `requirements-api.txt`）

## 快速启动

```bash
cd /Users/zzz/Desktop/Python/asr-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn api_server:app --host 127.0.0.1 --port 8000
