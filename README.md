# 原音 (Yuanyin)

原音是一个面向中文场景的语音复述产品原型：
- 主流程：语音输入 -> 文本识别 ->（可选克隆音色）-> 语音复述
- ASR：`best` 双引擎模式（Baidu + OpenAI 并行策略择优）
- TTS：ElevenLabs（克隆音色）/ OpenAI（普通复述）

## 当前能力

- 访客模式单页（默认开箱即用）
- 麦克风一键复述（开始录音 -> 结束并生成复述）
- 上传音频极速复述
- 候选确认回写（持续优化候选）
- 首次调用预热（显示 Baidu/OpenAI/ElevenLabs 可用性和延迟）
- 分段耗时展示（识别/克隆/复述/总耗时）
- 克隆音色缓存复用（同一 `Profile ID` 后续请求跳过重复克隆）

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

为了“最高准确率”建议同时填写两组 ASR（`best` 模式会自动择优）：

1. 中文优先
- `BAIDU_API_KEY`
- `BAIDU_SECRET_KEY`

2. 高鲁棒兜底
- `OPENAI_API_KEY`

语音复述相关：
- `ELEVENLABS_API_KEY`（可选；没有时会自动走 OpenAI 普通复述）

## 核心环境变量

- `ASR_PROVIDER_DEFAULT=best`（`best` / `auto` / `baidu` / `openai`）
- `ASR_LANGUAGE_HINT=zh`
- `BAIDU_DEV_PID=80001`
- `BAIDU_ASR_URL=https://vop.baidu.com/pro_api`
- `OPENAI_ASR_MODEL=gpt-4o-transcribe`

完整变量见 `/Users/zzz/Desktop/Python/asr-mvp/.env.example`。

## 常见问题

1. 识别慢
- 先点页面里的“首次调用预热”
- 如果你追求速度优先，可改为 `ASR_PROVIDER_DEFAULT=auto`
- 如果你追求准确率优先，保持 `ASR_PROVIDER_DEFAULT=best`

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
