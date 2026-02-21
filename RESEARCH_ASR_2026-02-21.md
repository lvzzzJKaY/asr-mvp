# 调研结论（2026-02-21）

目标：找“中文构音障碍语音识别”准确率更高的现有大模型，并用于本项目 API 接入。

## 1. 公开证据结论

1. 学术上，当前最有针对性的中文构音障碍基准是 **CDSD**（Interspeech 2024）。
- 参考：CDSD 数据集论文（Interspeech 2024）
- 链接：https://www.isca-archive.org/interspeech_2024/wu24f_interspeech.html

2. 面向构音障碍的“最高准确率”方案主要是 **Whisper 系列 + 说话人自适应微调**（例如 Perceiver-Prompt）。
- 该类工作在 CDSD 上报告了相对明显的 CER 改善（论文给出相对/绝对下降）。
- 参考（预印本页面）：
- 链接：https://www.academia.edu/126532016/Perceiver_Prompt_Tuning_A_Speaker_Aware_Unified_Prompting_Framework_for_Dysarthric_Speech_Recognition

3. 商用 API 侧：目前没有“统一、公开、同条件”的中文构音障碍 API 排行榜（可直接比较百度/OpenAI/讯飞/腾讯等在 CDSD 上的准确率）。
- 因此，无法严谨地仅凭公开资料给出“唯一第一名 API”。

## 2. 工程决策（用于本项目）

为了在真实可用的 API 条件下尽量接近“最高准确率”，采用：

- **ASR：best 双引擎策略**（Baidu + OpenAI）
  - 同时调用两路 ASR，按历史纠错匹配度 + 中文文本特征自动择优。
  - 好处：对构音障碍输入更稳，避免单一 API 失误。

- **克隆音色 + 复述：ElevenLabs（主）+ OpenAI（兜底）**
  - ElevenLabs 负责音色克隆与复述；当套餐/权限受限时自动降级到 OpenAI 普通复述。
  - 并新增“已克隆音色缓存复用”，避免每次重复克隆，显著降低后续时延。

## 3. 相关官方文档（API可用性与规格）

- Qwen-ASR-Realtime 规格与价格（阿里云官方）
  - https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-pricing-and-specifications
- OpenAI Speech-to-Text（官方）
  - https://platform.openai.com/docs/guides/speech-to-text
- ElevenLabs Voice Cloning（官方）
  - https://elevenlabs.io/docs/product-guides/voice-cloning
- 讯飞语音听写 API（官方）
  - https://www.xfyun.cn/doc/asr/voicedictation/API.html
- 腾讯云 ASR 文档（官方）
  - https://www.tencentcloud.com/document/product/1118/53937

## 4. 风险与下一步

- 如果你要求“严格可发表级别”的最高准确率，需要：
  1) 用你的真实样本构建标注集；
  2) 对候选 API 跑同一套离线评测（CER/WER）；
  3) 按结果确定最终单引擎或继续双引擎。

本仓库当前实现的是“在商用 API 约束下，准确率优先”的上线可用方案。
