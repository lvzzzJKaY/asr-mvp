#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import (
    AsrBackend,
    TrainingItem,
    build_hotword_prompt,
    load_profile,
    now_iso,
    normalize_text,
    profile_path,
    save_profile,
    score_training_text,
    suggest_candidates,
    update_hotwords,
)


app = FastAPI(
    title="Personalized Dysarthria ASR MVP API",
    description="将 CLI 版本的个性化 ASR MVP 封装为 HTTP 接口",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WEB_INDEX = BASE_DIR / "web" / "index.html"


class TrainRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    phrases: List[str] = Field(..., min_length=1)
    scene: Optional[str] = None


class TranscribeRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    audio_path: Optional[str] = None
    mock_text: Optional[str] = None
    model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    device: str = "cpu"
    topk: int = Field(3, ge=1, le=10)


class ConfirmRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    recognized: str = Field(..., min_length=1)
    chosen: str = Field(..., min_length=1)


class PhraseboardAddRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    scene: str = "custom"
    text: str = Field(..., min_length=1)


class ExportRequest(BaseModel):
    profile_id: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": now_iso()}


@app.get("/", include_in_schema=False)
def web_home() -> FileResponse:
    if not WEB_INDEX.exists():
        raise HTTPException(status_code=404, detail=f"前端页面不存在: {WEB_INDEX}")
    return FileResponse(WEB_INDEX)


@app.post("/train")
def train(req: TrainRequest) -> Dict:
    profile = load_profile(req.profile_id)
    added = []
    for phrase in req.phrases:
        norm = normalize_text(phrase)
        if not norm:
            continue
        score, reason = score_training_text(norm)
        profile.training_data.append(
            TrainingItem(
                text=norm,
                quality_score=score,
                quality_reason=reason,
                created_at=now_iso(),
            )
        )
        update_hotwords(profile, norm, delta=2 if score >= 0.9 else 1)
        if req.scene:
            profile.phraseboard.setdefault(req.scene, [])
            if norm not in profile.phraseboard[req.scene]:
                profile.phraseboard[req.scene].append(norm)
        added.append(
            {
                "text": norm,
                "quality_score": score,
                "quality_reason": reason,
                "suggest_re_record": score < 0.7,
            }
        )
    save_profile(profile)
    return {
        "profile_id": profile.profile_id,
        "added_count": len(added),
        "items": added,
    }


def _do_transcribe(
    profile_id: str,
    mock_text: Optional[str],
    audio_path: Optional[str],
    model: str,
    device: str,
    topk: int,
) -> Dict:
    profile = load_profile(profile_id)
    hotword = build_hotword_prompt(profile)
    if mock_text:
        text = normalize_text(mock_text)
    else:
        if not audio_path:
            raise HTTPException(status_code=400, detail="audio_path 与 mock_text 至少提供一个")
        p = Path(audio_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"音频不存在: {audio_path}")
        try:
            backend = AsrBackend(model=model, device=device, enable_vad=False, enable_punc=False)
            text = backend.transcribe(audio_path=str(p), hotword=hotword)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"ASR 推理失败: {exc}") from exc

    candidates = suggest_candidates(text, profile, top_k=topk)
    return {
        "profile_id": profile.profile_id,
        "recognized_text": text,
        "candidates": candidates,
        "hotword_count": len(profile.hotwords),
        "timestamp": now_iso(),
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest) -> Dict:
    return _do_transcribe(
        profile_id=req.profile_id,
        mock_text=req.mock_text,
        audio_path=req.audio_path,
        model=req.model,
        device=req.device,
        topk=req.topk,
    )


@app.post("/transcribe-upload")
async def transcribe_upload(
    profile_id: str,
    file: UploadFile = File(...),
    model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    device: str = "cpu",
    topk: int = 3,
) -> Dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名")
    suffix = Path(file.filename).suffix or ".wav"
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    with NamedTemporaryFile(prefix="asr_upload_", suffix=suffix, delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        return await run_in_threadpool(
            _do_transcribe,
            profile_id,
            None,
            tmp.name,
            model,
            device,
            topk,
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
    update_hotwords(profile, chosen, delta=2)
    save_profile(profile)
    return {"status": "ok", "profile_id": profile.profile_id, "chosen": chosen}


@app.get("/phraseboard")
def phraseboard(profile_id: str, scene: Optional[str] = None) -> Dict:
    profile = load_profile(profile_id)
    if scene:
        return {scene: profile.phraseboard.get(scene, [])}
    return profile.phraseboard


@app.post("/phraseboard/add")
def phraseboard_add(req: PhraseboardAddRequest) -> Dict:
    profile = load_profile(req.profile_id)
    scene = normalize_text(req.scene) or "custom"
    text = normalize_text(req.text)
    profile.phraseboard.setdefault(scene, [])
    if text not in profile.phraseboard[scene]:
        profile.phraseboard[scene].append(text)
        update_hotwords(profile, text, delta=1)
        save_profile(profile)
        return {"status": "added", "scene": scene, "text": text}
    return {"status": "exists", "scene": scene, "text": text}


@app.post("/export")
def export_profile(req: ExportRequest) -> Dict[str, str]:
    profile = load_profile(req.profile_id)
    out = Path(req.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(profile.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": "ok", "output": str(out)}


@app.delete("/profile/{profile_id}")
def delete_profile(profile_id: str) -> Dict[str, str]:
    p = profile_path(profile_id)
    if p.exists():
        p.unlink()
        return {"status": "deleted", "profile_id": profile_id}
    return {"status": "not_found", "profile_id": profile_id}
