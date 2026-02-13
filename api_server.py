#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import load_profile, now_iso, normalize_text, save_profile


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ASR_MODEL = os.getenv("OPENAI_ASR_MODEL", "whisper-1")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

# Chinese-first ASR provider (Baidu Speech REST API)
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "")
BAIDU_ASR_URL = os.getenv("BAIDU_ASR_URL", "https://vop.baidu.com/pro_api").rstrip("/")
BAIDU_DEV_PID = int(os.getenv("BAIDU_DEV_PID", "80001"))
BAIDU_CUID = os.getenv("BAIDU_CUID", "yuanyin_web")
BAIDU_OAUTH_URL = "https://aip.baidubce.com/oauth/2.0/token"

ASR_PROVIDER_DEFAULT = os.getenv("ASR_PROVIDER_DEFAULT", "auto").strip().lower()
ASR_LANGUAGE_HINT = os.getenv("ASR_LANGUAGE_HINT", "zh").strip().lower()

ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_CLONE_DISABLED_REASON: Optional[str] = None


def build_http_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


OPENAI_SESSION = build_http_session()
ELEVENLABS_SESSION = build_http_session()
BAIDU_SESSION = build_http_session()

BAIDU_ACCESS_TOKEN = ""
BAIDU_TOKEN_EXPIRES_AT = 0.0

BASE_DIR = Path(__file__).resolve().parent
WEB_INDEX = BASE_DIR / "web" / "index.html"


app = FastAPI(
    title="原音 API",
    description="原音：线上 ASR 与语音复述网关",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    mock_text: Optional[str] = None
    provider: str = "auto"
    topk: int = Field(3, ge=1, le=10)


class ConfirmRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    recognized: str = Field(..., min_length=1)
    chosen: str = Field(..., min_length=1)


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    provider: str = "elevenlabs"
    # ElevenLabs clone voice id
    voice_id: Optional[str] = None
    # For OpenAI/macOS fallback
    voice: Optional[str] = None
    model_id: Optional[str] = None
    rate: int = Field(180, ge=120, le=280)


def require_key(name: str, value: str) -> None:
    if not value:
        raise HTTPException(status_code=500, detail=f"缺少环境变量: {name}")


def is_instant_clone_plan_blocked(detail: str) -> bool:
    text = normalize_text(detail).lower()
    return "can_not_use_instant_voice_cloning" in text


def asr_language_value() -> str:
    value = normalize_text(ASR_LANGUAGE_HINT).lower()
    if not value or value == "auto":
        return ""
    return value


def warmup_openai() -> Dict[str, object]:
    if not OPENAI_API_KEY:
        return {"configured": False, "ok": False, "detail": "missing OPENAI_API_KEY"}
    url = f"{OPENAI_BASE_URL}/v1/models"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    start = time.perf_counter()
    try:
        resp = OPENAI_SESSION.get(url, headers=headers, timeout=15)
        latency_ms = int((time.perf_counter() - start) * 1000)
        detail = "ok" if resp.status_code < 400 else normalize_text(resp.text)[:180]
        return {
            "configured": True,
            "ok": resp.status_code < 400,
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
            "detail": detail,
        }
    except requests.RequestException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "configured": True,
            "ok": False,
            "latency_ms": latency_ms,
            "detail": f"network error: {exc}",
        }


def baidu_audio_format(filename: str, content_type: str) -> str:
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext in {"pcm", "wav", "amr", "m4a"}:
        return ext
    ctype = (content_type or "").lower()
    if "wav" in ctype:
        return "wav"
    if "amr" in ctype:
        return "amr"
    if "m4a" in ctype or "mp4" in ctype:
        return "m4a"
    if "pcm" in ctype:
        return "pcm"
    return ""


def baidu_access_token() -> str:
    global BAIDU_ACCESS_TOKEN, BAIDU_TOKEN_EXPIRES_AT
    if BAIDU_ACCESS_TOKEN and time.time() < BAIDU_TOKEN_EXPIRES_AT:
        return BAIDU_ACCESS_TOKEN
    require_key("BAIDU_API_KEY", BAIDU_API_KEY)
    require_key("BAIDU_SECRET_KEY", BAIDU_SECRET_KEY)
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY,
    }
    try:
        resp = BAIDU_SESSION.post(BAIDU_OAUTH_URL, params=params, timeout=15)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Baidu 鉴权网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"Baidu 鉴权失败: {resp.text}")
    try:
        body = resp.json()
        token = normalize_text(str(body.get("access_token", "")))
        expires_in = int(body.get("expires_in", 0))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Baidu 鉴权响应解析失败: {exc}") from exc
    if not token:
        raise HTTPException(status_code=500, detail=f"Baidu 鉴权未返回 token: {body}")
    BAIDU_ACCESS_TOKEN = token
    BAIDU_TOKEN_EXPIRES_AT = time.time() + max(60, expires_in - 60)
    return token


def warmup_baidu() -> Dict[str, object]:
    if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
        return {"configured": False, "ok": False, "detail": "missing BAIDU_API_KEY / BAIDU_SECRET_KEY"}
    start = time.perf_counter()
    try:
        token = baidu_access_token()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "configured": True,
            "ok": bool(token),
            "latency_ms": latency_ms,
            "detail": "ok",
        }
    except HTTPException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "configured": True,
            "ok": False,
            "latency_ms": latency_ms,
            "detail": normalize_text(str(exc.detail)),
        }


def warmup_elevenlabs() -> Dict[str, object]:
    if not ELEVENLABS_API_KEY:
        return {"configured": False, "ok": False, "detail": "missing ELEVENLABS_API_KEY"}
    url = f"{ELEVENLABS_BASE_URL}/v1/user"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    start = time.perf_counter()
    try:
        resp = ELEVENLABS_SESSION.get(url, headers=headers, timeout=15)
        latency_ms = int((time.perf_counter() - start) * 1000)
        detail = "ok" if resp.status_code < 400 else normalize_text(resp.text)[:180]
        return {
            "configured": True,
            "ok": resp.status_code < 400,
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
            "detail": detail,
        }
    except requests.RequestException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "configured": True,
            "ok": False,
            "latency_ms": latency_ms,
            "detail": f"network error: {exc}",
        }


def suggest_candidates_by_history(text: str, profile_id: str, topk: int) -> List[str]:
    profile = load_profile(profile_id)
    src = normalize_text(text)
    candidates = [src] if src else []
    ranked: List[tuple[float, str]] = []
    for item in profile.corrections[-120:]:
        chosen = normalize_text(item.get("chosen", ""))
        if not chosen:
            continue
        score = SequenceMatcher(None, src, chosen).ratio()
        if src and (src in chosen or chosen in src):
            score += 0.2
        ranked.append((score, chosen))
    ranked.sort(key=lambda x: x[0], reverse=True)
    for score, cand in ranked:
        if score < 0.35:
            continue
        if cand not in candidates:
            candidates.append(cand)
        if len(candidates) >= topk:
            break
    return candidates[:topk] if candidates else []


def asr_openai(file_bytes: bytes, filename: str, content_type: str) -> str:
    require_key("OPENAI_API_KEY", OPENAI_API_KEY)
    url = f"{OPENAI_BASE_URL}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
    data = {"model": OPENAI_ASR_MODEL}
    lang = asr_language_value()
    if lang:
        data["language"] = lang
    try:
        resp = OPENAI_SESSION.post(url, headers=headers, files=files, data=data, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI ASR 网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"OpenAI ASR 失败: {resp.text}")
    try:
        body = resp.json()
        text = normalize_text(str(body.get("text", "")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"OpenAI ASR 响应解析失败: {exc}") from exc
    if not text:
        raise HTTPException(status_code=500, detail="OpenAI ASR 返回空文本")
    return text


def asr_baidu(file_bytes: bytes, filename: str, content_type: str) -> str:
    token = baidu_access_token()
    fmt = baidu_audio_format(filename, content_type)
    if not fmt:
        raise HTTPException(
            status_code=400,
            detail="Baidu ASR 暂不支持该音频格式，请使用 wav/pcm/amr/m4a，或改走 OpenAI 兜底。",
        )
    payload = {
        "format": fmt,
        "rate": 16000,
        "channel": 1,
        "cuid": BAIDU_CUID,
        "token": token,
        "dev_pid": BAIDU_DEV_PID,
        "len": len(file_bytes),
        "speech": base64.b64encode(file_bytes).decode("ascii"),
    }
    try:
        resp = BAIDU_SESSION.post(BAIDU_ASR_URL, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Baidu ASR 网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"Baidu ASR 失败: {resp.text}")
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Baidu ASR 响应解析失败: {exc}") from exc
    err_no = int(body.get("err_no", -1))
    if err_no != 0:
        raise HTTPException(
            status_code=502,
            detail=f"Baidu ASR 失败 err_no={err_no}, err_msg={body.get('err_msg')}, sn={body.get('sn')}",
        )
    result = body.get("result") or []
    text = normalize_text(result[0] if result else "")
    if not text:
        raise HTTPException(status_code=500, detail="Baidu ASR 返回空文本")
    return text


def resolve_asr_provider(raw: str) -> str:
    provider = (raw or ASR_PROVIDER_DEFAULT or "auto").strip().lower()
    if provider not in {"auto", "openai", "baidu"}:
        raise HTTPException(status_code=400, detail=f"不支持的 ASR provider: {provider}")
    return provider


def asr_auto(file_bytes: bytes, filename: str, content_type: str) -> tuple[str, str]:
    errors: List[str] = []
    if BAIDU_API_KEY and BAIDU_SECRET_KEY:
        try:
            return asr_baidu(file_bytes, filename, content_type), "baidu"
        except HTTPException as exc:
            if exc.status_code != 400:
                errors.append(f"baidu:{exc.status_code}")
    if OPENAI_API_KEY:
        try:
            return asr_openai(file_bytes, filename, content_type), "openai"
        except HTTPException as exc:
            errors.append(f"openai:{exc.status_code}")
    if not BAIDU_API_KEY and not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="缺少 ASR API 配置: BAIDU_API_KEY 或 OPENAI_API_KEY")
    raise HTTPException(status_code=502, detail=f"自动 ASR 失败: {'; '.join(errors) or '未知错误'}")


def asr_by_provider(provider: str, file_bytes: bytes, filename: str, content_type: str) -> tuple[str, str]:
    chosen = resolve_asr_provider(provider)
    if chosen == "openai":
        return asr_openai(file_bytes, filename, content_type), "openai"
    if chosen == "baidu":
        return asr_baidu(file_bytes, filename, content_type), "baidu"
    return asr_auto(file_bytes, filename, content_type)


def clone_voice_elevenlabs_parts(
    name: str,
    file_parts: List[tuple[str, tuple[str, bytes, str]]],
    description: str,
) -> Dict:
    require_key("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not file_parts:
        raise HTTPException(status_code=400, detail="样本音频为空")
    url = f"{ELEVENLABS_BASE_URL}/v1/voices/add"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    data = {"name": name}
    if description:
        data["description"] = description
    try:
        resp = ELEVENLABS_SESSION.post(url, headers=headers, data=data, files=file_parts, timeout=300)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs 克隆网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs 克隆失败: {resp.text}")
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ElevenLabs 克隆响应解析失败: {exc}") from exc


def clone_voice_elevenlabs(name: str, files: List[UploadFile], description: str) -> Dict:
    require_key("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个样本音频")
    file_parts: List[tuple[str, tuple[str, bytes, str]]] = []
    for f in files:
        content = f.file.read()
        if not content:
            continue
        file_parts.append(
            (
                "files",
                (
                    f.filename or "sample.wav",
                    content,
                    f.content_type or "application/octet-stream",
                ),
            )
        )
    return clone_voice_elevenlabs_parts(name=name, file_parts=file_parts, description=description)


def tts_elevenlabs_audio(text: str, voice_id: str, model_id: Optional[str]) -> bytes:
    require_key("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not voice_id:
        raise HTTPException(status_code=400, detail="ElevenLabs 需要 voice_id")
    url = f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    params = {"output_format": "mp3_44100_128"}
    payload = {"text": text, "model_id": model_id or ELEVENLABS_TTS_MODEL}
    try:
        resp = ELEVENLABS_SESSION.post(url, headers=headers, params=params, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs TTS 网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs TTS 失败: {resp.text}")
    return resp.content


def tts_elevenlabs(text: str, voice_id: str, model_id: Optional[str]) -> Response:
    audio = tts_elevenlabs_audio(text=text, voice_id=voice_id, model_id=model_id)
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
    )


def tts_openai_audio(text: str, voice: Optional[str], model_id: Optional[str]) -> bytes:
    require_key("OPENAI_API_KEY", OPENAI_API_KEY)
    url = f"{OPENAI_BASE_URL}/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id or OPENAI_TTS_MODEL,
        "voice": voice or OPENAI_TTS_VOICE,
        "input": text,
        "format": "mp3",
    }
    try:
        resp = OPENAI_SESSION.post(url, headers=headers, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI TTS 网络错误: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"OpenAI TTS 失败: {resp.text}")
    return resp.content


def tts_openai(text: str, voice: Optional[str], model_id: Optional[str]) -> Response:
    audio = tts_openai_audio(text=text, voice=voice, model_id=model_id)
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
    )


def tts_macos_say(text: str, voice: Optional[str], rate: int) -> Response:
    if shutil.which("say") is None:
        raise HTTPException(status_code=500, detail="系统未找到 say 命令")
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(prefix="tts_", suffix=".aiff", delete=False) as tmp_aiff:
        aiff_path = Path(tmp_aiff.name)
    wav_path = aiff_path.with_suffix(".wav")

    def _run_say(v: Optional[str]) -> None:
        cmd = ["say", "-o", str(aiff_path), "-r", str(rate)]
        if v:
            cmd.extend(["-v", v])
        cmd.append(text)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        try:
            _run_say(voice)
        except subprocess.CalledProcessError:
            _run_say(None)
        if shutil.which("afconvert") is not None:
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            data = wav_path.read_bytes()
            media_type = "audio/wav"
            filename = "speech.wav"
        else:
            data = aiff_path.read_bytes()
            media_type = "audio/aiff"
            filename = "speech.aiff"
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"macOS 语音合成失败: {exc.stderr or exc}") from exc
    finally:
        if aiff_path.exists():
            aiff_path.unlink()
        if wav_path.exists():
            wav_path.unlink()
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": now_iso()}


@app.get("/config")
def config() -> Dict[str, object]:
    try:
        asr_default = resolve_asr_provider(ASR_PROVIDER_DEFAULT)
    except HTTPException:
        asr_default = "auto"
    clone_status = "ready"
    if ELEVENLABS_CLONE_DISABLED_REASON:
        clone_status = "disabled_cached"
    elif not ELEVENLABS_API_KEY:
        clone_status = "no_key"
    return {
        "asr_provider_default": asr_default,
        "asr_language_hint": asr_language_value() or "auto",
        "tts_provider_default": "elevenlabs",
        "openai_asr_model": OPENAI_ASR_MODEL,
        "baidu_dev_pid": BAIDU_DEV_PID,
        "openai_tts_model": OPENAI_TTS_MODEL,
        "elevenlabs_clone_status": clone_status,
        "elevenlabs_clone_note": ELEVENLABS_CLONE_DISABLED_REASON or "",
        "online_ready": {
            "openai": bool(OPENAI_API_KEY),
            "baidu": bool(BAIDU_API_KEY and BAIDU_SECRET_KEY),
            "elevenlabs": bool(ELEVENLABS_API_KEY),
        },
    }


@app.post("/warmup")
def warmup() -> Dict[str, object]:
    openai = warmup_openai()
    baidu = warmup_baidu()
    elevenlabs = warmup_elevenlabs()
    overall_ok = bool(openai.get("ok")) or bool(baidu.get("ok")) or bool(elevenlabs.get("ok"))
    return {
        "status": "ok" if overall_ok else "partial",
        "providers": {
            "openai": openai,
            "baidu": baidu,
            "elevenlabs": elevenlabs,
        },
        "timestamp": now_iso(),
    }


@app.get("/", include_in_schema=False)
def web_home() -> FileResponse:
    if not WEB_INDEX.exists():
        raise HTTPException(status_code=404, detail=f"前端页面不存在: {WEB_INDEX}")
    return FileResponse(WEB_INDEX)


@app.post("/transcribe")
def transcribe(req: TranscribeRequest) -> Dict:
    text = normalize_text(req.mock_text or "")
    if not text:
        raise HTTPException(status_code=400, detail="当前接口仅用于 mock 调试，请使用 /transcribe-upload 上传音频")
    candidates = suggest_candidates_by_history(text, req.profile_id, req.topk)
    return {
        "profile_id": req.profile_id,
        "provider": req.provider,
        "recognized_text": text,
        "candidates": candidates if candidates else [text],
        "timestamp": now_iso(),
    }


@app.post("/transcribe-upload")
async def transcribe_upload(
    profile_id: str,
    file: UploadFile = File(...),
    provider: str = ASR_PROVIDER_DEFAULT,
    topk: int = 3,
) -> Dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    text, provider_used = asr_by_provider(provider, payload, file.filename, file.content_type or "")
    candidates = suggest_candidates_by_history(text, profile_id, topk)
    return {
        "profile_id": profile_id,
        "provider": provider_used,
        "recognized_text": text,
        "candidates": candidates if candidates else [text],
        "timestamp": now_iso(),
    }


@app.post("/transcribe-clone-speak")
async def transcribe_clone_speak(
    profile_id: str = Form(...),
    file: UploadFile = File(...),
    asr_provider: str = Form(ASR_PROVIDER_DEFAULT),
    tts_provider: str = Form("auto"),
    allow_fallback: bool = Form(True),
    topk: int = Form(1),
    clone_name: str = Form(""),
    clone_description: str = Form(""),
) -> Dict:
    global ELEVENLABS_CLONE_DISABLED_REASON
    t0 = time.perf_counter()
    asr_ms = 0
    clone_ms = 0
    tts_ms = 0

    def timing_dict() -> Dict[str, int]:
        total_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "asr": int(asr_ms),
            "clone": int(clone_ms),
            "tts": int(tts_ms),
            "total": total_ms,
        }

    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")

    asr_start = time.perf_counter()
    text, asr_provider_used = asr_by_provider(asr_provider, payload, file.filename, file.content_type or "")
    asr_ms = int((time.perf_counter() - asr_start) * 1000)
    candidates = suggest_candidates_by_history(text, profile_id, topk)
    selected_text = candidates[0] if candidates else text

    tts_provider = tts_provider.lower().strip()
    if tts_provider not in {"auto", "elevenlabs", "openai"}:
        raise HTTPException(status_code=400, detail=f"不支持的 TTS provider: {tts_provider}")

    base_result = {
        "profile_id": profile_id,
        "asr_provider_used": asr_provider_used,
        "recognized_text": text,
        "candidates": candidates if candidates else [text],
        "selected_text": selected_text,
        "timestamp": now_iso(),
    }

    def build_openai_result(mode: str, note: str = "") -> Dict:
        nonlocal tts_ms
        tts_start = time.perf_counter()
        speech_bytes = tts_openai_audio(text=selected_text, voice=None, model_id=None)
        tts_ms = int((time.perf_counter() - tts_start) * 1000)
        return {
            **base_result,
            "mode": mode,
            "note": note,
            "clone_name": "",
            "voice_id": "",
            "speech_mime": "audio/mpeg",
            "speech_base64": base64.b64encode(speech_bytes).decode("ascii"),
            "timing_ms": timing_dict(),
        }

    if tts_provider == "openai":
        return build_openai_result(mode="openai_tts")
    if tts_provider == "auto" and ELEVENLABS_CLONE_DISABLED_REASON:
        return build_openai_result(
            mode="fallback_openai_tts_cached",
            note=f"已启用极速模式：跳过克隆，直接普通复述。原因: {ELEVENLABS_CLONE_DISABLED_REASON}",
        )

    normalized_name = normalize_text(clone_name)
    final_name = normalized_name or f"asr-echo-{profile_id}-{int(time.time())}"
    try:
        clone_start = time.perf_counter()
        clone_result = clone_voice_elevenlabs_parts(
            name=final_name,
            description=normalize_text(clone_description),
            file_parts=[
                (
                    "files",
                    (
                        file.filename,
                        payload,
                        file.content_type or "application/octet-stream",
                    ),
                )
            ],
        )
        clone_ms = int((time.perf_counter() - clone_start) * 1000)
        voice_id = normalize_text(str(clone_result.get("voice_id", "")))
        if not voice_id:
            raise HTTPException(status_code=500, detail="克隆成功但未返回 voice_id")
        tts_start = time.perf_counter()
        speech_bytes = tts_elevenlabs_audio(text=selected_text, voice_id=voice_id, model_id=None)
        tts_ms = int((time.perf_counter() - tts_start) * 1000)
        return {
            **base_result,
            "mode": "cloned_elevenlabs",
            "note": "",
            "clone_name": final_name,
            "voice_id": voice_id,
            "speech_mime": "audio/mpeg",
            "speech_base64": base64.b64encode(speech_bytes).decode("ascii"),
            "timing_ms": timing_dict(),
        }
    except HTTPException as exc:
        if is_instant_clone_plan_blocked(str(exc.detail)):
            ELEVENLABS_CLONE_DISABLED_REASON = str(exc.detail)
        if tts_provider == "elevenlabs" or not allow_fallback:
            raise
        return build_openai_result(
            mode="fallback_openai_tts",
            note=f"ElevenLabs 克隆不可用，已自动切换 OpenAI 普通复述。原因: {exc.detail}",
        )


@app.post("/confirm")
def confirm(req: ConfirmRequest) -> Dict[str, str]:
    profile = load_profile(req.profile_id)
    recognized = normalize_text(req.recognized)
    chosen = normalize_text(req.chosen)
    profile.corrections.append(
        {
            "recognized": recognized,
            "chosen": chosen,
            "created_at": now_iso(),
        }
    )
    save_profile(profile)
    return {"status": "ok", "profile_id": profile.profile_id, "chosen": chosen}


@app.post("/voice-clone")
async def voice_clone(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...),
) -> Dict:
    name = normalize_text(name)
    if not name:
        raise HTTPException(status_code=400, detail="音色名称不能为空")
    return clone_voice_elevenlabs(name=name, files=files, description=description)


@app.post("/speak")
def speak(req: SpeakRequest) -> Response:
    text = normalize_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    provider = req.provider.lower().strip()
    if provider == "elevenlabs":
        return tts_elevenlabs(text=text, voice_id=req.voice_id or "", model_id=req.model_id)
    if provider == "openai":
        return tts_openai(text=text, voice=req.voice, model_id=req.model_id)
    if provider == "macos":
        return tts_macos_say(text=text, voice=req.voice, rate=req.rate)
    raise HTTPException(status_code=400, detail=f"不支持的 TTS provider: {provider}")
