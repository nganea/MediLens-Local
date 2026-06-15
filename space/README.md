---
title: MediLens
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
short_description: Offline, multilingual medicine-label helper
tags:
  - track:backyard
  - sponsor:openbmb
  - sponsor:openai
  - achievement:offgrid
  - achievement:offbrand
  - achievement:llama
---

# MediLens: Medicine Label Helper

MediLens helps someone read a medicine label and understand, in plain language
and in their own language, what the medicine is commonly used for.

This Space runs the **full MediLens app** exactly as designed. Because a free
Hugging Face Space has **no GPU**, the two local AI models and the robot cannot
run here:

- **MiniCPM-V 4.6** (OpenBMB) - vision OCR that reads labels from a photo.
- **Tiny Aya Global (3B)** (Cohere / Cohere Labs) - rewrites and translates the
  explanation at a 14-15 year-old reading level.
- **Reachy Mini (Hugging Face)** - a hands-free, multilingual voice assistant
  (offline speech via faster-whisper / Whisper, Kokoro, and Piper).

What works fully on this hosted demo: **medicine lookup** against a local
200-medicine database and **offline multilingual explanations** (English,
French, German, Italian, Romanian, Spanish). The AI-model and Reachy controls
are shown so the interface matches the desktop app; a note in the app explains
they need a local GPU.

To experience everything (vision OCR, AI translation, and the Reachy Mini robot),
run MediLens on your own computer from the GitHub repository, or watch the demo
video.

## Links

- **Demo video:** https://youtube.com/shorts/nUSK8DPznm4
- **GitHub repository:** https://github.com/nganea/MediLens-Local
- **LinkedIn post:** https://www.linkedin.com/posts/natasa-ganea_medilens-offline-ai-medicine-helper-with-share-7472309911658610689-yFtq/

## Safety

Informational only. MediLens does not give dosage instructions, does not tell
anyone to take a medicine, and does not confirm a medicine is safe for a
specific person. Always check with a pharmacist or doctor.

## Credits

Models: **MiniCPM-V 4.6** by OpenBMB and **Tiny Aya Global** by Cohere / Cohere
Labs. Robot: **Reachy Mini** by Hugging Face. Offline speech: faster-whisper
(OpenAI Whisper), Kokoro, and Piper. Local text OCR uses Tesseract; name
matching uses rapidfuzz.

This app was developed with the help of **OpenAI Codex** and **Anthropic Claude**.
