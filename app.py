#!/usr/bin/env python3
"""
构音障碍者个性化 ASR MVP（CLI 版）

实现调研报告中的关键任务：
1) 个性化训练入口（短句录入 + 质量检测）
2) 个人词库/热词偏置
3) 识别结果候选确认并回写学习
4) 常用短语板（场景模板）
5) 本地存储、导出、删除（隐私可控）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple


DATA_DIR = Path("profiles")

DEFAULT_PHRASEBOARD = {
    "medical": [
        "我现在头疼，想预约明天下午门诊",
        "请帮我联系家属",
        "我对青霉素过敏",
        "请再说一遍，我需要确认药名",
    ],
    "travel": [
        "请带我去这个地址",
        "我已经到医院门口",
        "请确认目的地和价格",
        "请在前面路口停车",
    ],
    "family": [
        "我现在不舒服，需要帮助",
        "我已到达，请放心",
        "请帮我发一条微信消息",
        "我想休息十分钟",
    ],
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def char_valid_ratio(text: str) -> float:
    if not text:
        return 0.0
    valid = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum() or ch in "，。,.!?！？:：;；- ":
            valid += 1
    return valid / max(1, len(text))


@dataclass
class TrainingItem:
    text: str
    quality_score: float
    quality_reason: str
    created_at: str


@dataclass
class Profile:
    profile_id: str
    created_at: str
    updated_at: str
    hotwords: Dict[str, int] = field(default_factory=dict)
    phraseboard: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_PHRASEBOARD))
    training_data: List[TrainingItem] = field(default_factory=list)
    corrections: List[Dict[str, str]] = field(default_factory=list)

    def to_json(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hotwords": self.hotwords,
            "phraseboard": self.phraseboard,
            "training_data": [item.__dict__ for item in self.training_data],
            "corrections": self.corrections,
        }

    @staticmethod
    def from_json(data: Dict) -> "Profile":
        training_items = []
        for item in data.get("training_data", []):
            training_items.append(
                TrainingItem(
                    text=item.get("text", ""),
                    quality_score=float(item.get("quality_score", 0)),
                    quality_reason=item.get("quality_reason", ""),
                    created_at=item.get("created_at", now_iso()),
                )
            )
        return Profile(
            profile_id=data["profile_id"],
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            hotwords={k: int(v) for k, v in data.get("hotwords", {}).items()},
            phraseboard={
                k: [normalize_text(x) for x in v if normalize_text(x)]
                for k, v in data.get("phraseboard", {}).items()
            },
            training_data=training_items,
            corrections=data.get("corrections", []),
        )


def profile_path(profile_id: str) -> Path:
    return DATA_DIR / f"{profile_id}.json"


def load_profile(profile_id: str) -> Profile:
    p = profile_path(profile_id)
    if not p.exists():
        created = now_iso()
        profile = Profile(profile_id=profile_id, created_at=created, updated_at=created)
        save_profile(profile)
        return profile
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Profile.from_json(data)


def save_profile(profile: Profile) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    profile.updated_at = now_iso()
    with profile_path(profile.profile_id).open("w", encoding="utf-8") as f:
        json.dump(profile.to_json(), f, ensure_ascii=False, indent=2)


def score_training_text(text: str) -> Tuple[float, str]:
    t = normalize_text(text)
    if len(t) < 4:
        return 0.2, "文本过短，建议重录（至少 4 个字符）"
    if len(t) > 60:
        return 0.6, "文本偏长，建议拆分为更短短句"
    ratio = char_valid_ratio(t)
    if ratio < 0.7:
        return 0.4, "有效字符占比低，建议重录或清理噪声字符"
    unique_ratio = len(set(t)) / max(1, len(t))
    if unique_ratio < 0.25:
        return 0.5, "重复字符较多，建议更自然表达"
    return 0.95, "质量良好"


def update_hotwords(profile: Profile, phrase: str, delta: int = 1) -> None:
    phrase = normalize_text(phrase)
    if not phrase:
        return
    profile.hotwords[phrase] = max(1, profile.hotwords.get(phrase, 0) + delta)


def build_hotword_prompt(profile: Profile, cap: int = 80) -> str:
    pairs = sorted(profile.hotwords.items(), key=lambda kv: kv[1], reverse=True)
    words = []
    for phrase, weight in pairs[:cap]:
        repeat = 1 if weight <= 1 else (2 if weight < 4 else 3)
        words.extend([phrase] * repeat)
    return " ".join(words).strip()


def phrase_pool(profile: Profile) -> List[str]:
    all_phrases = []
    for items in profile.phraseboard.values():
        all_phrases.extend(items)
    all_phrases.extend(profile.hotwords.keys())
    seen = set()
    uniq = []
    for p in all_phrases:
        t = normalize_text(p)
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def ensure_ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        local_bin = Path(__file__).resolve().parent / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        link_path = local_bin / "ffmpeg"
        if not link_path.exists():
            try:
                link_path.symlink_to(ffmpeg_path)
            except OSError:
                link_path.write_text(f"#!/bin/sh\nexec \"{ffmpeg_path}\" \"$@\"\n", encoding="utf-8")
                link_path.chmod(0o755)
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def suggest_candidates(raw_text: str, profile: Profile, top_k: int = 3) -> List[str]:
    text = normalize_text(raw_text)
    ranked: List[Tuple[float, str]] = []
    for phrase in phrase_pool(profile):
        score = SequenceMatcher(None, text, phrase).ratio()
        if phrase in text or text in phrase:
            score += 0.2
        ranked.append((score, phrase))
    ranked.sort(key=lambda x: x[0], reverse=True)
    candidates = [text] if text else []
    for _, cand in ranked:
        if cand not in candidates:
            candidates.append(cand)
        if len(candidates) >= top_k:
            break
    return candidates[:top_k]


class AsrBackend:
    _MODEL_CACHE = {}

    def __init__(self, model: str, device: str, enable_vad: bool = False, enable_punc: bool = False) -> None:
        self.model_name = model
        self.device = device
        self.enable_vad = enable_vad
        self.enable_punc = enable_punc
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        cache_key = (self.model_name, self.device, self.enable_vad, self.enable_punc)
        if cache_key in AsrBackend._MODEL_CACHE:
            self._model = AsrBackend._MODEL_CACHE[cache_key]
            return self._model
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "无法导入 funasr。请先安装：pip install -U funasr，"
                f"或在仓库内运行并确保依赖完整。原始错误：{exc}"
            ) from exc
        kwargs = {"model": self.model_name, "device": self.device}
        if self.enable_vad:
            kwargs["vad_model"] = "fsmn-vad"
        if self.enable_punc:
            kwargs["punc_model"] = "ct-punc"
        self._model = AutoModel(**kwargs)
        AsrBackend._MODEL_CACHE[cache_key] = self._model
        return self._model

    def transcribe(self, audio_path: str, hotword: str) -> str:
        model = self._ensure_model()
        input_data = audio_path
        # Prefer loading wav directly to avoid requiring ffmpeg in minimal setups.
        try:
            import soundfile as sf  # type: ignore

            speech, _ = sf.read(audio_path)
            if hasattr(speech, "ndim") and speech.ndim > 1:
                speech = speech.mean(axis=1)
            input_data = [speech]
        except Exception:
            ensure_ffmpeg_available()
        res = model.generate(input=input_data, batch_size_s=60, hotword=hotword)
        if not res:
            return ""
        first = res[0] if isinstance(res, list) else res
        if isinstance(first, dict):
            return normalize_text(str(first.get("text", "")))
        return normalize_text(str(first))


def cmd_train(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    added = []
    for phrase in args.phrase:
        norm = normalize_text(phrase)
        if not norm:
            continue
        score, reason = score_training_text(norm)
        item = TrainingItem(text=norm, quality_score=score, quality_reason=reason, created_at=now_iso())
        profile.training_data.append(item)
        update_hotwords(profile, norm, delta=2 if score >= 0.9 else 1)
        if args.scene:
            profile.phraseboard.setdefault(args.scene, [])
            if norm not in profile.phraseboard[args.scene]:
                profile.phraseboard[args.scene].append(norm)
        added.append(item)
    save_profile(profile)

    print(f"Profile: {profile.profile_id}")
    print(f"新增训练短语: {len(added)}")
    for item in added:
        advice = "建议重录" if item.quality_score < 0.7 else "可用"
        print(f"- {item.text} | 质量={item.quality_score:.2f} | {item.quality_reason} | {advice}")


def cmd_transcribe(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    hotword = build_hotword_prompt(profile)
    if args.mock_text:
        text = normalize_text(args.mock_text)
    else:
        if not Path(args.audio).exists():
            raise FileNotFoundError(f"音频不存在: {args.audio}")
        backend = AsrBackend(model=args.model, device=args.device)
        text = backend.transcribe(audio_path=args.audio, hotword=hotword)

    candidates = suggest_candidates(text, profile, top_k=args.topk)
    result = {
        "profile_id": profile.profile_id,
        "recognized_text": text,
        "candidates": candidates,
        "hotword_count": len(profile.hotwords),
        "timestamp": now_iso(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_confirm(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    recognized = normalize_text(args.recognized)
    chosen = normalize_text(args.chosen)
    profile.corrections.append(
        {
            "recognized": recognized,
            "chosen": chosen,
            "created_at": now_iso(),
        }
    )
    update_hotwords(profile, chosen, delta=2)
    save_profile(profile)
    print(f"已确认候选并回写: {chosen}")


def cmd_phraseboard(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    if args.add:
        text = normalize_text(args.add)
        scene = args.scene or "custom"
        profile.phraseboard.setdefault(scene, [])
        if text and text not in profile.phraseboard[scene]:
            profile.phraseboard[scene].append(text)
            update_hotwords(profile, text, delta=1)
            save_profile(profile)
            print(f"已添加到短语板[{scene}]: {text}")
        else:
            print("短语为空或已存在，无需重复添加。")
        return

    if args.scene:
        scenes = {args.scene: profile.phraseboard.get(args.scene, [])}
    else:
        scenes = profile.phraseboard
    print(json.dumps(scenes, ensure_ascii=False, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    profile = load_profile(args.profile)
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(profile.to_json(), f, ensure_ascii=False, indent=2)
    print(f"已导出: {out}")


def cmd_delete(args: argparse.Namespace) -> None:
    p = profile_path(args.profile)
    if p.exists():
        p.unlink()
        print(f"已删除 profile: {args.profile}")
    else:
        print(f"profile 不存在: {args.profile}")


def cmd_reset(args: argparse.Namespace) -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"已清空本地数据目录: {DATA_DIR.resolve()}")
    else:
        print(f"数据目录不存在: {DATA_DIR.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构音障碍者个性化 ASR MVP CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="录入训练短句并做质量检测")
    p_train.add_argument("--profile", required=True, help="用户 profile id")
    p_train.add_argument("--phrase", action="append", required=True, help="训练短句，可重复传入")
    p_train.add_argument("--scene", help="可选：同时加入某个场景短语板")
    p_train.set_defaults(func=cmd_train)

    p_trans = sub.add_parser("transcribe", help="识别音频并生成候选")
    p_trans.add_argument("--profile", required=True, help="用户 profile id")
    p_trans.add_argument("--audio", default="", help="音频路径（不使用 --mock-text 时必填）")
    p_trans.add_argument("--model", default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", help="FunASR 模型名")
    p_trans.add_argument("--device", default="cpu", help="推理设备：cpu/cuda")
    p_trans.add_argument("--topk", type=int, default=3, help="候选数")
    p_trans.add_argument("--mock-text", help="演示模式：跳过 ASR，直接用该文本走候选流程")
    p_trans.set_defaults(func=cmd_transcribe)

    p_confirm = sub.add_parser("confirm", help="确认候选并回写学习")
    p_confirm.add_argument("--profile", required=True, help="用户 profile id")
    p_confirm.add_argument("--recognized", required=True, help="原识别文本")
    p_confirm.add_argument("--chosen", required=True, help="用户确认文本")
    p_confirm.set_defaults(func=cmd_confirm)

    p_phrase = sub.add_parser("phraseboard", help="查看或新增短语板短句")
    p_phrase.add_argument("--profile", required=True, help="用户 profile id")
    p_phrase.add_argument("--scene", help="场景名，不传则查看全部")
    p_phrase.add_argument("--add", help="新增短语")
    p_phrase.set_defaults(func=cmd_phraseboard)

    p_export = sub.add_parser("export", help="导出 profile（本地隐私可导出）")
    p_export.add_argument("--profile", required=True, help="用户 profile id")
    p_export.add_argument("--output", required=True, help="导出 JSON 路径")
    p_export.set_defaults(func=cmd_export)

    p_delete = sub.add_parser("delete", help="删除单个 profile（本地隐私可删除）")
    p_delete.add_argument("--profile", required=True, help="用户 profile id")
    p_delete.set_defaults(func=cmd_delete)

    p_reset = sub.add_parser("reset-all", help="清空全部本地数据")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
