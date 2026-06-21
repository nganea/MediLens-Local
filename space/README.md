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

> **About this page.** This is the Hugging Face Space card for the hosted,
> CPU-only demo. A free Hugging Face Space has no GPU, so the two local AI
> models (MiniCPM-V 4.6 vision OCR and Tiny Aya Global) and the Reachy Mini
> robot do not run here. On the hosted Space, medicine lookup against the local
> 200-medicine database and offline multilingual explanations work fully; the
> AI-model and Reachy controls are shown so the interface matches the desktop
> app. **To run the full app locally (vision OCR, AI translation, Reachy Mini),
> follow the local install guide below.**

## Links

- **Demo video:** https://youtube.com/shorts/nUSK8DPznm4
- **GitHub repository:** https://github.com/nganea/MediLens-Local
- **LinkedIn post:** https://www.linkedin.com/posts/natasa-ganea_medilens-offline-ai-medicine-helper-with-share-7472309911658610689-yFtq/

---

MediLens Local is a small local-first Gradio app that reads a medicine label image with OCR, fuzzy-matches the text against a local CSV medicine database, and shows a plain-language explanation of what the medicine is commonly used for.

It does not use cloud APIs and does not replace a pharmacist, doctor, or other qualified healthcare professional.

This app was developed with the help of OpenAI Codex and Claude Opus.

## Models and Credits

MediLens runs two small local models through `llama.cpp`, both used fully offline:

- **MiniCPM-V 4.6** (vision OCR) - by **OpenBMB** (`openbmb/MiniCPM-V-4.6-gguf` on Hugging Face). Reads medicine label text from a photo.
- **Tiny Aya Global** (3B, explanation and translation) - by **Cohere / Cohere Labs** (`CohereLabs/tiny-aya-global-GGUF` on Hugging Face). Rewrites and translates the plain-language explanation.

Local text OCR also uses **Tesseract** (`pytesseract`), and name matching uses **rapidfuzz**.

The **Reachy Mini** voice assistant adds an offline speech stack:

- **Speech-to-text:** **`faster-whisper`** (a fast CTranslate2 reimplementation of **OpenAI Whisper**) - `base.en` for English and the multilingual `small` model for other languages.
- **Text-to-speech (English):** **Kokoro** neural voice via `kokoro-onnx` (by hexgrad).
- **Text-to-speech (Romanian, German, French, Italian, Spanish):** **Piper** voices (by the Open Home Foundation / Rhasspy): `ro_RO-mihai`, `de_DE-thorsten`, `fr_FR-siwis`, `it_IT-paola`, `es_ES-davefx`.
- **Fallback voice:** Windows **SAPI** (Microsoft).

The app was developed with the help of **OpenAI Codex** and **Claude Opus**.

## Features

- Upload or capture a medicine label image.
- Extract text locally with Tesseract OCR through `pytesseract`.
- Optional local vision-model OCR with MiniCPM-V 4.6 GGUF.
- Match messy OCR text to local medicine names and brand names with `rapidfuzz`.
- Show extracted text, likely match, confidence, common uses, and safety warnings.
- Handle unclear labels with a low-confidence message.
- Template responses in English, French, German, Italian, Spanish, and Romanian.
- Optional local multilingual explanation and translation with the Tiny Aya Global GGUF model.
- "Improve translation" section for submitting reviewed glossary suggestions.
- Reusable `medilens_core` package for desktop, API, or robot adapters.
- Reachy Mini hands-free, multilingual voice assistant (confirmed working on hardware), startable from the desktop app.

## Files

- `app.py` - Gradio desktop application, UI event wiring, and Reachy Mini start/stop controls.
- `medilens_core/` - Reusable medicine database, matching, OCR, model-server, explanation, and robot pipeline code.
- `static/medilens.css` - Gradio UI styling.
- `robot/medilens_robot_service.py` - Local HTTP service (`/health`, `/status`, `/identify`) the robot calls.
- `robot/reachy_mini_app.py` - Reachy Mini adapter and CLI: camera capture, speech, head/antenna motion, hands-free listening.
- `robot/voice_listener.py` - Offline microphone listener (energy VAD + faster-whisper).
- `medicines.csv` - Local 200-row starter medicine database.
- `translation_glossary.csv` - Reviewed translation glossary used to improve Tiny Aya phrasing.
- `space/` - Self-contained lightweight build for Hugging Face Spaces (CPU-only demo).
- `requirements.txt` - Python package dependencies.
- `README.md` - Setup and usage notes.

## Setup

1. Install Python 3.10 or newer.
2. Install Tesseract OCR on your computer.

   Windows: install Tesseract from the UB Mannheim build, then make sure `tesseract.exe` is on your PATH.

   macOS:

   ```bash
   brew install tesseract
   ```

   Ubuntu or Debian:

   ```bash
   sudo apt-get install tesseract-ocr
   ```

3. Install `llama.cpp` for the two local AI models:

   ```bash
   winget install llama.cpp
   ```

   After installing, close Git Bash and open it again so the `llama-server` command is available.

   Use a **recent** `llama.cpp` build. MiniCPM-V is a vision model and needs its
   multimodal projector (`mmproj`); recent builds download and load it
   automatically with `-hf`, while older builds run text-only and fail to read
   images. If you already have `llama.cpp`, run `winget upgrade llama.cpp`. See
   the MiniCPM troubleshooting section below if image reading does not work.

   To use a GPU-enabled `llama-server`, set GPU layers before starting the app:

   ```bash
   export MEDILENS_LLAMA_GPU_LAYERS=99
   ```

   On PowerShell, use:

   ```powershell
   $env:MEDILENS_LLAMA_GPU_LAYERS="99"
   ```

   When this is set, MediLens starts `llama-server` with `-ngl 99`.

4. Install Python dependencies:

   ```bash
   cd ~/Documents/Codex/MediLens
   py -m pip install -r requirements.txt
   ```

## Start Everything In Git Bash

You normally need **three Git Bash windows**:

- one for Tiny Aya
- one for MiniCPM-V
- one for the Gradio app

Keep all three windows open while using the app.

### Window 1: Start Tiny Aya

Open Git Bash and run:

```bash
llama-server -hf CohereLabs/tiny-aya-global-GGUF:Q4_K_M --port 8080
```

For GPU mode, add `-ngl 99`:

```bash
llama-server -hf CohereLabs/tiny-aya-global-GGUF:Q4_K_M --port 8080 -ngl 99
```

Wait until you see something like:

```text
server is listening on http://127.0.0.1:8080
```

Leave this window open.

### Window 2: Start MiniCPM-V

Open a second Git Bash window and run:

```bash
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081
```

For GPU mode, add `-ngl 99`:

```bash
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081 -ngl 99
```

Wait until you see something like:

```text
server is listening on http://127.0.0.1:8081
```

Leave this window open.

### Window 3: Start The App

Open a third Git Bash window and run:

   ```bash
   cd ~/Documents/Codex/MediLens
   py app.py
   ```

Wait until you see:

```text
Running on local URL: http://127.0.0.1:7860
```

Open this address in your browser:

```text
http://127.0.0.1:7860
```

## Reachy Mini Voice Assistant

The reusable medicine logic lives in `medilens_core`, so Reachy Mini does not run the Gradio web UI. Instead, Reachy Mini is a **hands-free, multilingual voice assistant** that has been confirmed working on real hardware. Everything runs offline.

The flow:

1. Reachy loads and calibrates its microphone, waggles its antennas, and says the listening prompt.
2. The wake cue is Reachy's name plus a medicine word.
3. Reachy says "Okay", captures the label with its camera, does a thinking head motion, identifies the medicine, and speaks the answer in the same language it detected.
4. It keeps listening for the next question until you say "Reachy stop" or press Stop in the desktop app.

The speech stack (all offline):

- **Text-to-speech:** Kokoro neural voice (English) and Piper voices (Romanian, German, French, Italian, Spanish), with Windows SAPI as a fallback.
- **Speech-to-text:** `faster-whisper` (English-only `base.en`, multilingual `small` for other languages).
- Listening can use the laptop microphone or Reachy Mini's own microphone (the desktop button defaults to Reachy's mic).

### Easiest way to start it: the desktop app

In **Technical details** in the desktop app there is a **Start Reachy Mini MediLens** / **Stop Reachy Mini MediLens** button. Before starting, turn on Reachy Mini at `reachy-mini.local:8000`, make sure other robot apps are off, and allow Windows Firewall if prompted. The button starts the local robot service and the hands-free app in the background, and feeds Reachy's result back into the desktop fields. Stop gracefully shuts Reachy down (goodbye phrase, antennas lowered, process exit).

### Manual / command-line use

The components are:

- `robot/medilens_robot_service.py` - local HTTP service (`/health`, `/status`, `/identify`).
- `robot/reachy_mini_app.py` - Reachy adapter and CLI.
- `robot/voice_listener.py` - offline mic listener.

Start MiniCPM-V 4.6 and the robot service on the laptop/desktop:

```bash
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081
python robot/medilens_robot_service.py --host 0.0.0.0 --port 8765
```

Make sure Reachy Mini and the laptop are on the same Wi-Fi network and the laptop firewall allows inbound connections on port `8765`. You can sanity-check the service from the same laptop at `http://127.0.0.1:8765/health`.

Run the hands-free assistant (this is effectively what the desktop button runs):

```bash
py -3 robot/reachy_mini_app.py --use-reachy --listen --language auto \
  --orientation-mode "Normal first" --max-vision-attempts 1 \
  --service-url http://127.0.0.1:8765 --reachy-host reachy-mini.local \
  --reachy-port 8000 --timeout 90 --mic reachy --show-result
```

You can also test the pipeline with a plain image file, no robot required:

```bash
python robot/reachy_mini_app.py path/to/medicine-label.jpg --service-url http://127.0.0.1:8765 --timeout 90
```

### Use The Camera

The image box supports both:

- uploading an existing image
- using the camera option to capture a new photo

On the same computer as the app, open:

```text
http://127.0.0.1:7860
```

Then use the camera option in the image box. Your browser may ask for camera permission.

### Speed Up Image Reading

In **Technical details**, the **Image orientation mode** setting controls how many image orientations MiniCPM tries.

- `Normal first` is best for uploaded images, phone/tablet camera images, and captures where the label text is already readable.
- `Mirrored first` is best for desktop or laptop webcam captures where the label appears left-right reversed.
- `Full auto` is the broadest fallback, but it can be slow because MiniCPM may need to try several transformed images.

On page load the orientation mode defaults to `Normal first`, which suits uploaded images and phone/tablet photos. Choose `Full auto` only for desktop webcam captures, where the label may appear reversed or rotated. The desktop default is `2` attempts per orientation for caution; the Reachy Mini path uses `1` attempt for speed.

The app does not start local AI models just because the page loads. It checks/starts the local model servers when you upload/capture an image or type a medicine name and click **Search**. While models are being checked or started, the Search button is disabled and the text under it reads "Loading AI models...". In **Technical details**, **Max image processing time, seconds** lets the user choose a wait time from 10 to 99 seconds (default 60). If MiniCPM cannot identify the image with high confidence within that time, the app stops and asks the user to type the medicine name or label text instead.

During MiniCPM image reading, the app shows small status text under the **Search** button (for example `Processing image... 18/60s`). The reserved status area stays in place when processing finishes, so the fields do not shift.

### Use A Phone Camera

The simplest phone workflow is:

1. Make sure the phone and computer are on the same Wi-Fi network.
2. Start the app so other devices on your network can reach it.

   Git Bash:

   ```bash
   export MEDILENS_SERVER_NAME=0.0.0.0
   py app.py
   ```

   PowerShell:

   ```powershell
   $env:MEDILENS_SERVER_NAME="0.0.0.0"
   py app.py
   ```

3. Find your computer's local network IP address.

   PowerShell:

   ```powershell
   ipconfig
   ```

4. On the phone, open this address, replacing the example IP address with your computer's IP address:

   ```text
   http://<LAPTOP_IP>:7860
   ```

5. In the image box, choose upload. On most phones, the upload picker lets you take a new photo with the phone camera.

Live webcam capture on a phone browser may be blocked on a plain `http://192.168...` address because mobile browsers often require HTTPS for direct camera access. If you need the live camera button on a phone, you can start Gradio with a temporary HTTPS share link:

Git Bash:

```bash
export MEDILENS_SHARE=1
py app.py
```

PowerShell:

```powershell
$env:MEDILENS_SHARE="1"
py app.py
```

This creates a temporary Gradio share URL. The app logic and models still run on your computer, but the page is reachable through Gradio's share service, so use this only if that is acceptable for your testing.

### If A Command Is Not Found

If `py` is not found, try:

```bash
python app.py
```

If `llama-server` is not found after installing `llama.cpp`, close Git Bash and open it again. If it still is not found, restart Windows.

## Optional Local Tiny Aya Global GGUF Model

The app can use `CohereLabs/tiny-aya-global-GGUF` to rewrite and translate the plain-language explanation. This is optional. The OCR and matching workflow still works without it.

The app does not call a cloud API. It talks to a local `llama-server` running on your computer.

The recommended file is the 4-bit quantized model:

```text
tiny-aya-global-q4_k_m.gguf
```

Install `llama.cpp` on Windows:

```bash
winget install llama.cpp
```

Start Tiny Aya Global in a separate terminal:

```bash
llama-server -hf CohereLabs/tiny-aya-global-GGUF:Q4_K_M --port 8080
```

The first run downloads the model. Later runs use the local cache.

Then start the Gradio app in another terminal:

```bash
py app.py
```

The app calls this local URL by default:

```text
http://127.0.0.1:8080/v1/chat/completions
```

You can set a different local server URL before starting the app:

Git Bash:

```bash
export TINY_AYA_MODEL_URL="http://127.0.0.1:8080/v1/chat/completions"
py app.py
```

PowerShell:

```powershell
$env:TINY_AYA_MODEL_URL="http://127.0.0.1:8080/v1/chat/completions"
py app.py
```

Then tick **Use local Tiny Aya Global server for explanation** in the app.

If the local server is not running, the app falls back to the simple template explanation and shows the connection error in the explanation box.

Tiny Aya is a 3B parameter model family. The Hugging Face GGUF page lists `tiny-aya-global-q4_k_m.gguf` as about 2.14 GB, which is a practical first choice for a local app. A larger BF16 or F16 file may be higher quality but needs much more memory.

## Optional Local MiniCPM-V 4.6

If Tesseract struggles with stylized medicine logos, you can run a local vision-language OCR server with MiniCPM-V 4.6 GGUF.

Start MiniCPM-V in a separate terminal:

```bash
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081
```

The first run downloads the model. Later runs use the local cache.

Then start the Gradio app in another terminal:

```bash
py app.py
```

In the app, tick:

```text
Use local MiniCPM-V 4.6 server
```

The app calls this local URL by default:

```text
http://127.0.0.1:8081/v1/chat/completions
```

You can keep Tiny Aya on port `8080` for explanation and MiniCPM-V on port `8081` for OCR.

If a local model server is not running, the app can try to start `llama-server` in the background. Use the **Check/retry local model servers** button in the app, or let the app attempt a background start after a connection failure. The first start can take a minute or more while the model loads. If the app says it cannot find `llama-server`, restart your terminal after installing `llama.cpp` or start the server manually.

To start background model servers with GPU layers, set this before launching the app:

```bash
export MEDILENS_LLAMA_GPU_LAYERS=99
```

The app will add `-ngl 99` to each `llama-server` command. Leave the variable unset, or set it to `0`, for CPU mode.

If the MiniCPM-V server is not running and cannot be started, the app shows Tesseract OCR text for debugging but does not trust it for medicine matching.

### Troubleshooting: MiniCPM does not read the image (but Tesseract does)

MiniCPM-V is a **vision** model, so `llama.cpp` needs two files to read images:

- the main model GGUF (downloaded by `-hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M`), and
- the **multimodal projector** (`mmproj`) GGUF, which is the model's "eyes".

If `llama-server` starts with only the main GGUF, it runs fine as a **text-only**
server: the port is reachable and the app thinks MiniCPM is "running", but every
image request fails. The app then silently falls back to Tesseract OCR. The
symptom is exactly: **Tesseract reads the image, MiniCPM does not.**

There are two reasons the projector may be absent at runtime: it was **never
downloaded**, or it was **downloaded but not loaded** by `llama-server`. Start by
checking whether the file is on disk:

```bash
find ~/.cache/huggingface -iname "*mmproj*"
```

The projector lives next to the main GGUF under
`~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6-gguf/`, named
`mmproj-model-f16.gguf` (f16 is the only published precision for the projector,
which is normal even when the main model is `Q4_K_M`).

**Case A - the `find` command prints nothing (mmproj not downloaded).** The
projector was never fetched. Re-pull the model with a recent `llama.cpp`, which
downloads both files:

```bash
winget upgrade llama.cpp
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081
```

Run the `find` command again to confirm `mmproj-model-f16.gguf` now exists.

**Case B - the `find` command lists `mmproj-model-f16.gguf` (downloaded but not
loaded).** The file is present but `llama-server` is ignoring it. **Recent
`llama.cpp` builds load the `mmproj` automatically with `-hf`, but older builds do
not**, so the file sits on disk unused and the server runs **text-only**. This is
the most common cause on a freshly set up computer.

Either way, confirm by watching the MiniCPM `llama-server` startup log: a
vision-capable server prints a line that loads `mmproj-model-f16.gguf` (often
mentioning `clip` or `vision`). If you see no such line, vision is not loaded.

For Case B, first update `llama.cpp`, then close and reopen Git Bash and start
MiniCPM again:

```bash
winget upgrade llama.cpp
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --port 8081
```

If the log now shows the `mmproj` load line, vision is working.

If your build still will not load it, point `llama-server` at both files
explicitly with `-m` and `--mmproj` (this works on older builds too). The glob
expands to the snapshot folder in your Hugging Face cache:

```bash
llama-server \
  -m ~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6-gguf/snapshots/*/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj ~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6-gguf/snapshots/*/mmproj-model-f16.gguf \
  --port 8081
```

Finally, make sure **Use local MiniCPM-V 4.6 model** is ticked in **Technical
details** in the app. If it is unticked, the app uses Tesseract only.

## Languages and Translation

The app supports English, French, German, Italian, Romanian, and Spanish. The language dropdown is alphabetized, and the app can pick an initial language from the browser setting while staying local. When the language changes, the field labels and page text update and existing results are cleared, so press **Search** again.

Tiny Aya rewrites the medicine use and safety warning to be easier to read (aimed at a 14-15 year-old reading level, at most three short sentences, no dosage instructions) and translates them into the selected language.

### Improve translation

Above **Technical details** there is an **Improve translation** section where users can submit a glossary suggestion (language, bad phrase, preferred phrase). Suggestions are saved to `translation_glossary_suggestions.csv` and are **not** applied automatically - they need human review before being moved into the reviewed `translation_glossary.csv`.

## Hugging Face Space

A self-contained, CPU-only build for Hugging Face Spaces lives in `space/`. It keeps the parts that run reliably on free CPU (manual name entry, optional Tesseract OCR, local database matching, offline multilingual template explanations) and drops the local llama.cpp servers and Reachy integration, which are shown in the demo video. To deploy, create a Gradio Space and upload the contents of `space/` (`app.py`, `README.md`, `requirements.txt`, `packages.txt`, `medicines.csv`) at the repo root.

## How It Works

1. The user uploads or captures an image.
2. `pytesseract` extracts text from the image.
3. The app cleans the OCR text.
4. Each medicine generic name and brand name in `medicines.csv` is compared with the OCR text.
5. The highest fuzzy-match score is selected.
6. Confidence is classified as:

- high: score >= 90
- medium: score 80 to 89
- low: score < 80

If confidence is low, the app asks the user to try a clearer photo or ask a pharmacist.

## Safety

The app always includes this warning:

> Do not take this medicine unless it was prescribed for you. If you are unsure what this medicine is or whether it is safe for you, ask a pharmacist, doctor, or other qualified healthcare professional.

The app only explains what a medicine is commonly used for. It does not tell anyone to take a medicine, does not confirm that a medicine is safe for a person, and does not provide dosage instructions.

## Expanding the Medicine Database

The project now includes a local 200-row starter database covering common medicines across UK, US, and European usage patterns. It is intended for local label matching and hackathon prototyping, not as a certified formulary.

The rows are marked `needs_review` because medicine data should be reviewed against local clinical sources before public or clinical use. Normal app use does not call RxNorm, NHS, dm+d, or any other medicine API.

Add or update rows in `medicines.csv` using this format:

```csv
generic_name,brand_names,common_uses,safety_warning,source_name,source_url,last_checked,review_status,notes
examplegeneric,BrandOne;BrandTwo,a short plain-language description,Ask a pharmacist or doctor if unsure.,NHS Medicines A to Z,https://www.nhs.uk/medicines/examplegeneric/,2026-06-12,needs_review,Short note
```

Separate multiple brand names with semicolons.

The app only needs these four columns to run:

- `generic_name`
- `brand_names`
- `common_uses`
- `safety_warning`

The extra columns are there so the database can be maintained safely:

- `source_name` - where the common-use summary came from.
- `source_url` - the source page used for review.
- `last_checked` - date the row was reviewed.
- `review_status` - for example `starter_reviewed` or `needs_review`.
- `notes` - short internal notes.

Recommended workflow:

1. Use NHS Medicines A to Z for patient-friendly common-use summaries during manual review.
2. Use downloaded RxNorm or dm+d files to improve name matching and brand aliases if you need a fully offline source.
3. Keep `common_uses` as a short English summary.
4. Let Tiny Aya translate or rewrite the summary in the selected language.
5. Do not let an automated source update overwrite the reviewed CSV without a human check.

### Offline Source Notes

NHS website content is generally available under the Open Government Licence unless excluded, but adapted or translated clinical content may no longer carry NHS clinical approval. For this app, keep NHS-derived summaries short, source-linked, and reviewed.

RxNorm and dm+d are useful for names and identifiers. If you cannot use APIs, use their downloadable files or manually curated exports, then review changes before copying names into `medicines.csv`. They are not plain-language patient explanation sources.

### Source Consistency Check

The app keeps the active medicine database in `medicines.csv`. The only helper script kept in this repository is the current source consistency checker:

```bash
py scripts/check_semantic_consistency.py
```

It compares the `common_uses` and `safety_warning` columns in `medicines.csv` with each row's `source_url`, then writes `medicine_semantic_consistency_report.csv`.

Some source pages may block automated checks. If that happens, save copied page text in:

```text
source_pages/<generic-name>.txt
```

For example:

```text
source_pages/celecoxib.txt
```

Then rerun:

```bash
py scripts/check_semantic_consistency.py
```

The consistency report is a triage tool. A low score means "review this row"; it does not prove the row is wrong.
