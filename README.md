# MediLens Local: Medicine Label Helper

MediLens Local is a small local-first Gradio app that reads a medicine label image with OCR, fuzzy-matches the text against a local CSV medicine database, and shows a plain-language explanation of what the medicine is commonly used for.

It does not use cloud APIs and does not replace a pharmacist, doctor, or other qualified healthcare professional.

This app was developed with the help of OpenAI Codex and Claude.

## Features

- Upload or capture a medicine label image.
- Extract text locally with Tesseract OCR through `pytesseract`.
- Optional local vision-model OCR with MiniCPM-V 4.6 GGUF.
- Match messy OCR text to local medicine names and brand names with `rapidfuzz`.
- Show extracted text, likely match, confidence, common uses, and safety warnings.
- Handle unclear labels with a low-confidence message.
- Template responses in English, French, German, Italian, Spanish, and Romanian.
- Optional local multilingual explanation generation with the Tiny Aya Global GGUF model.
- Reusable `medilens_core` package for desktop, API, or robot adapters.
- Reachy Mini adapter scaffold for English-only spoken medicine identification.

## Files

- `app.py` - Gradio desktop application and UI event wiring.
- `medilens_core/` - Reusable medicine database, matching, OCR, model-server, explanation, and robot pipeline code.
- `static/medilens.css` - Gradio UI styling.
- `robot/reachy_mini_app.py` - Reachy Mini adapter scaffold and image-file test entry point.
- `medicines.csv` - Local 200-row starter medicine database.
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

## Reachy Mini Prototype

The reusable medicine logic now lives in `medilens_core`, so Reachy Mini does not need to run the Gradio web UI. The recommended hackathon setup is:

- Run MiniCPM-V 4.6 on the desktop or laptop at `http://127.0.0.1:8081/v1/chat/completions`, or expose that address on the same local network for Reachy.
- Let Reachy Mini handle the cue phrase, camera capture, speech, and thinking head movement.
- Call `medilens_core.pipeline.identify_medicine_from_image(...)` with a 90-second timeout.
- Speak only the English result from the local medicine database.

The scaffold is in:

```bash
robot/reachy_mini_app.py
```

You can test the robot flow with an image file before wiring the real Reachy SDK:

```bash
python robot/reachy_mini_app.py path/to/medicine-label.jpg --timeout 90
```

Replace the methods on `ReachyMiniHooks` with the actual Reachy Mini SDK calls for:

- `speak`
- `capture_image`
- `start_thinking_motion`
- `stop_thinking_motion`

The intended spoken flow is:

1. User: "Hey Reachy Mini, what's this medicine for?"
2. Reachy: "Okay, hold the medicine label in front of me."
3. Reachy captures a photo.
4. Reachy: "I have taken a picture. I am checking it now."
5. Reachy moves its head while MiniCPM-V 4.6 and database matching run.
6. If found: "It looks like Paracetamol. It is used for..."
7. If not found: "I do not know what this medicine is. Try the MediLens app on your device."

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

The app tries to set this automatically:

- phone or tablet: `Normal first`
- desktop or laptop: `Full auto`

The app checks local model server status when you upload/capture an image or click read. It does not constantly check server status in the background. If reading takes a long time, it is usually because MiniCPM is processing one or more image orientations. In **Technical details**, **Max image processing time, seconds** lets the user choose a two-digit wait time from 10 to 99 seconds. The default is 60 seconds. If MiniCPM cannot identify the image with high confidence within that time, the app stops and asks the user to type the medicine name or label text instead.

During MiniCPM image reading, the app shows small status text under the **Read medicine label** button if reading takes more than 5 seconds. The reserved status area stays in place when processing finishes, so the fields do not shift.

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
   http://192.168.1.25:7860
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

If a local model server is not running, the app can try to start `llama-server` in the background. Use the **Start/check local model servers** button in the app, or let the app attempt a background start after a connection failure. The first start can take a minute or more while the model loads. If the app says it cannot find `llama-server`, restart your terminal after installing `llama.cpp` or start the server manually.

If the MiniCPM-V server is not running and cannot be started, the app shows Tesseract OCR text for debugging but does not trust it for medicine matching.

## How It Works

1. The user uploads or captures an image.
2. `pytesseract` extracts text from the image.
3. The app cleans the OCR text.
4. Each medicine generic name and brand name in `medicines.csv` is compared with the OCR text.
5. The highest fuzzy-match score is selected.
6. Confidence is classified as:

- high: score >= 85
- medium: score 70 to 84
- low: score < 70

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
