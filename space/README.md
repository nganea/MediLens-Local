---
title: MediLens Local
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Offline medicine-label helper using small local models
---

# MediLens Local: Medicine Label Helper

MediLens Local helps someone read a medicine label and understand, in plain
language and in their own language, what the medicine is commonly used for.

This Space is the **lightweight, CPU-friendly demo** of the project. Type a
medicine name (or upload a label photo) and pick a language; MediLens fuzzy-
matches the text against a local 200-medicine database and shows the likely
medicine, its common use, a safety warning, and the source - fully offline, no
cloud APIs.

## How it fits the Build Small Hackathon (Backyard AI)

The real user is a relative who does not read English medicine labels well. The
full desktop version uses two small local models through `llama.cpp`:

- **MiniCPM-V 4.6** (vision OCR) by **OpenBMB** - reads stylised medicine labels
  from a photo.
- **Tiny Aya Global (3B)** by **Cohere / Cohere Labs** - rewrites and translates
  the explanation into the user's language at a 14-15 year-old reading level.

It also drives a **Reachy Mini** robot for a hands-free, multilingual voice
assistant. Those parts need local model servers and a GPU, so they are shown in
the demo video. This hosted Space keeps the parts that run reliably on free CPU:
the local database matching and offline multilingual template explanations.

## Credits

Models: **MiniCPM-V 4.6** by OpenBMB and **Tiny Aya Global** by Cohere / Cohere
Labs. Local text OCR uses Tesseract; name matching uses rapidfuzz. The app was
developed with the help of **OpenAI Codex** and **Claude Opus**.

## Safety

Informational only. MediLens does not give dosage instructions, does not tell
anyone to take a medicine, and does not confirm a medicine is safe for a
specific person. Always check with a pharmacist or doctor.
