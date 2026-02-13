# 原音 (Yuanyin)

原音是一个面向中文场景的语音复述产品原型：
- 主流程：语音输入 -> 文本识别 ->（可选克隆音色）-> 语音复述
- ASR：Baidu 中文优先，OpenAI 自动兜底
- TTS：ElevenLabs（克隆音色）/ OpenAI（普通复述）

## 当前能力

- 访客模式单页（默认开箱即用）
- 麦克风一键复述（开始录音 -> 结束并生成复述）
- 上传音频极速复述
- 候选确认回写（持续优化候选）
- 首次调用预热（显示 Baidu/OpenAI/ElevenLabs 可用性和延迟）
- 分段耗时展示（识别/克隆/复述/总耗时）

## 本地启动

```bash
cd /Users/zzz/Desktop/Python/asr-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
cp -n .env.example .env
```

编辑 `.env` 后启动：

```bash
set -a; source .env; set +a
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

打开：`http://127.0.0.1:8000`

## 必填配置

至少填一组 ASR：

1. 中文优先（推荐）
- `BAIDU_API_KEY`
- `BAIDU_SECRET_KEY`

2. 兜底（可选但建议）
- `OPENAI_API_KEY`

语音复述相关：
- `ELEVENLABS_API_KEY`（可选；没有时会自动走 OpenAI 普通复述）

## 核心环境变量

- `ASR_PROVIDER_DEFAULT=auto`（`auto` / `baidu` / `openai`）
- `ASR_LANGUAGE_HINT=zh`
- `BAIDU_DEV_PID=80001`
- `BAIDU_ASR_URL=https://vop.baidu.com/pro_api`

完整变量见 `/Users/zzz/Desktop/Python/asr-mvp/.env.example`。

## 常见问题

1. 识别慢
- 先点页面里的“首次调用预热”
- 使用 `ASR_PROVIDER_DEFAULT=auto`，优先走 Baidu

2. ElevenLabs 报 instant voice cloning 权限不足
- 这是账号套餐限制，不是代码问题
- 系统会自动降级到 OpenAI 普通复述

3. 麦克风录音走不到 Baidu
- Baidu 对音频格式要求更严格（wav/pcm/amr/m4a）
- 浏览器录音通常是 webm，`auto` 模式会自动回退 OpenAI

## Android 打包

见 `/Users/zzz/Desktop/Python/asr-mvp/mobile/README_ANDROID.md`

## 部署

见 `/Users/zzz/Desktop/Python/asr-mvp/DEPLOY.md`
