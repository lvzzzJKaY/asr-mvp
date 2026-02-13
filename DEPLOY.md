# 原音部署说明

## 架构

- ASR：Baidu（中文优先）+ OpenAI（兜底）
- TTS/克隆：ElevenLabs（可选）+ OpenAI（兜底）
- 后端：FastAPI (`api_server.py`)

## 1) 配置环境变量

基于 `.env.example` 配置：

- `BAIDU_API_KEY`
- `BAIDU_SECRET_KEY`
- `ASR_PROVIDER_DEFAULT`：`auto` / `baidu` / `openai`
- `ASR_LANGUAGE_HINT`：默认 `zh`
- `OPENAI_API_KEY`（建议作为兜底）
- `ELEVENLABS_API_KEY`（可选）

## 2) Docker 运行

```bash
docker build -t yuanyin-api .
docker run --rm -p 8000:8000 --env-file .env yuanyin-api
```

## 3) Render / Railway / Fly.io

启动命令：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 4) 健康与预热

- 健康检查：`GET /health`
- 配置检查：`GET /config`
- 预热：`POST /warmup`

## 5) 接口清单

- `POST /transcribe-upload`
- `POST /transcribe-clone-speak`
- `POST /confirm`
- `POST /voice-clone`
- `POST /speak`

## 6) 线上注意事项

1. 如果是浏览器录音（webm），Baidu 可能不支持该格式，系统会自动兜底 OpenAI。  
2. ElevenLabs 无 Instant Voice Cloning 权限时，会自动降级普通复述。  
3. 建议先调用一次 `POST /warmup`，减少首个请求延迟。  
