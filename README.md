# STAI — Smart Threat & Activity Intelligence

STAI is a local Python CCTV analysis tool. It uses a small YOLO ONNX model through OpenCV DNN on the CPU, OpenCV for crowd-density overlays and camera-quality checks, and optional sampled Google Gemma vision review. It does not install PyTorch, Ultralytics, or CUDA. It is a prototype for human review, not a verified emergency detector.

## What this repository contains

- Webcam or video-file monitoring with person boxes, a grid heatmap, and configurable crowd thresholds. Use `--frame_stride 2` to reduce CPU load when processing video; some ONNX exports require 640px even if a smaller `--imgsz` is requested.
- Camera-quality warnings for invalid, very dark, blurry, obstructed, or repeatedly identical frames. A static view alone does not mark a feed frozen.
- Optional sampled Gemma analysis, run in a background worker when Google credentials are configured.
- Optional Hugging Face hosted action classification through Inference Providers. It is disabled by default and requires an `HF_TOKEN`.
- Sampled Gemma activity assessments validated against a strict schema, with SQLite incident history, duplicate suppression, review policy, and notification delivery status.
- Telegram delivery disabled by default. Normal activity, ordinary social interaction, and quality-only warnings do not qualify for emergency notices.

This checkout is a standalone Python application. It does not contain the React dashboard, FastAPI service, authentication, or camera management API described in some proposed STAI designs. The incident store and policy helpers are ready for later integration with a service/UI; the existing OpenCV window remains the live interface.

## Architecture

```text
Webcam / video file
       │
       ├── YOLOv8 person boxes ── counts and grid heatmap
       ├── OpenCV frame checks ── camera-quality status
       ├── Optional sampled Gemma review (Google API)
       └── Optional sampled clip analysis (Hugging Face Inference API)
                     │
              validated STAI core
         incident history and review policy
                     │
         optional, explicit Telegram delivery
```

Gemma descriptions are model assessments. A single frame cannot establish intent, emotion, consent, a confirmed injury, or a confirmed emergency. Possible falls and hazards need human verification. The tool does not contact emergency services or control equipment.

## Install

Python 3.10+ is recommended. OpenCV display support is needed for the live window. The local detector requires a compatible YOLO `.onnx` model file; `.pt` weights are not loaded by OpenCV DNN. Put the ONNX file in the project folder or provide its path with `--model`. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The default detector filename is `yolov8n.onnx`. OpenCV DNN runs it locally on CPU; no PyTorch runtime or CUDA libraries are installed. The existing `yolov8n.pt` file is a different format and cannot be used directly. Download the 12.8 MB ONNX model from the model repository, then run a webcam:

```bash
curl -L -o yolov8n.onnx https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx
python detect_final.py --video 0 --model ./yolov8n.onnx
```

Or process a video:

```bash
python detect_final.py --video ./sample.mp4 --model ./yolov8n.onnx --frame_stride 2
```

Press **Q** in the video window to quit. The live application needs a display and camera/video source.

No simulated detections or incidents are used by the live application. No evaluation accuracy is claimed. Evaluate the hosted model on labeled clips from the target cameras before enabling threshold-based decisions.

## Environment

Copy `.env.example` to `.env`. Credentials belong only in `.env`, which is ignored by Git.

| Variable | Default | Purpose |
|---|---:|---|
| `GEMINI_API_KEY` | empty | Enables optional Google GenAI/Gemma analysis when `--gemma` is passed |
| `HF_TOKEN` | empty | Hugging Face token with Inference Providers permission |
| `GEMMA_MODEL` | `gemma-4-26b-a4b-it` | Optional Gemma model ID (CLI `--gemma_model` also exists) |
| `GEMMA_INTERVAL_SEC` | mode-dependent | Optional frame-sampling default; CLI `--gemma_interval` is available |
| `GEMMA_MAX_REQUESTS_PER_MINUTE` | `6` | Caps Gemma sampling rate, even when a shorter interval is requested |
| `GEMMA_REQUEST_TIMEOUT_SEC` | `15` | Hosted vision request timeout |
| `GEMMA_MAX_ATTEMPTS` | `2` | Total request attempts, including the initial call |
| `TELEGRAM_ENABLED` | `false` | Must be explicitly enabled before any Telegram message is sent |
| `TELEGRAM_BOT_TOKEN` | empty | Telegram bot token; legacy `BOT_TOKEN` is accepted |
| `TELEGRAM_CHAT_ID` | empty | Destination chat; legacy `CHAT_ID` is accepted |
| `STAI_CAMERA_ID` | `camera-0` | Stable camera/location label for incident records |
| `STAI_INCIDENT_DB` | `stai_incidents.sqlite3` | Local SQLite incident history path |
| `STAI_DASHBOARD_URL` | empty | Optional base URL used to construct incident links in notifications |
| `STAI_BLUR_VARIANCE_MIN` | `35` | Camera blur threshold |
| `STAI_BRIGHTNESS_MIN` | `28` | Camera brightness threshold |
| `ANOMALY_DETECTION_ENABLED` | `false` | Sends sampled clip frames to Hugging Face for hosted vision analysis |
| `ANOMALY_MODEL_PROVIDER` | `huggingface_inference` | Hugging Face Inference Providers API |
| `ANOMALY_MODEL_NAME` | `prithivMLmods/Human-Action-Recognition` | Human action classifier model ID |
| `ANOMALY_CLIP_DURATION_SECONDS` | `5` | Duration of the sampling window; its middle frame is classified |
| `ANOMALY_FRAME_SAMPLING_INTERVAL_SECONDS` | `1` | Interval between sampled frames in each window |
| `ANOMALY_MIN_PERSISTENCE_WINDOWS` | `2` | Consecutive anomalous windows before an incident is grouped |
| `ANOMALY_ALERT_COOLDOWN_SECONDS` | `60` | Minimum time between anomaly Telegram notices |
| `ANOMALY_INFERENCE_TIMEOUT_SECONDS` | `45` | Maximum wait for each Hugging Face Inference request |

Gemma is optional. Install its separate SDK only when you need hosted vision: `python -m pip install google-genai`. Then set `GEMINI_API_KEY` and run with `--gemma` (or `--detector gemma`). Google API calls happen only when Gemma mode is enabled. Review Google API data handling and camera privacy requirements before sending frames to a hosted model.

Telegram notifications are off unless `TELEGRAM_ENABLED=true` (or `1`, `yes`, `on`) and both Telegram credentials are configured. Never commit `.env` or share bot tokens. Telegram requests have timeouts and failures do not stop video analysis.

## Activity, quality, and incidents

`stai_core.py` contains a strict activity assessment schema for `NORMAL`, `SOCIAL_INTERACTION`, `POSSIBLE_FALL`, `POSSIBLE_VIOLENCE`, `CROWD_SURGE`, `FIRE_OR_SMOKE`, `OTHER_HAZARD`, and `UNCERTAIN`. It validates risk, model confidence, evidence, review flag, and video-quality status. It also provides:

- frame-quality checks with configurable thresholds and a repeated-frame window;
- SQLite incident history with duplicate suppression;
- guarded transitions among `NEW`, `UNDER_REVIEW`, `ACKNOWLEDGED`, `RESOLVED`, and `FALSE_ALARM`;
- a review policy that stays quiet for normal/social activity and quality-only warnings;
- plain-text Telegram message construction that labels confidence as model confidence.

When Gemma mode is enabled, the sampled frame is assessed with the structured activity schema and stored in SQLite. A qualifying high or critical possible hazard may trigger a Telegram human-review notice when delivery is enabled. The incident store can update review states from Python, but this checkout does not expose those controls through a web API or searchable dashboard. Likewise, temporal fall/violence action recognition and motion tracking are not claimed to be implemented by a single sampled VLM frame.

`anomaly_detection.py` sends one representative real camera/video frame from each sampling window to Hugging Face's image-classification API for `prithivMLmods/Human-Action-Recognition`. It labels supported classes such as `WORKING` (the model's `using_laptop` class), `EATING`, `FIGHTING`, `HUGGING`, `SITTING`, and `TEXTING`; the overlay includes classifier confidence. This is single-image classification, not temporal action recognition or per-person tracking. A Hugging Face token with Inference Providers permission is required. Free accounts have a limited monthly credit allowance; requests stop with a clear error when the account has no credit. Camera frames leave the machine and are submitted to Hugging Face. The linked demo Space is an OAuth front end, not a public `/predict` endpoint, so STAI calls the inference API directly. Set `ANOMALY_DETECTION_ENABLED=true` to enable it.

## Configuration options

```bash
python detect_final.py --help
```

Important options include `--video`, `--model`, `--conf`, `--grid`, `--hot`, `--max_people`, `--detector yolo|gemma`, `--gemma`, `--gemma_model`, and `--gemma_interval`. YOLO remains the default local person detector. Gemma mode samples frames and is slower and less consistent; it is not a real-time safety guarantee.

`--imgsz` must be a multiple of 32. Many pretrained ONNX exports use a fixed 640×640 input; STAI retries at 640 if a smaller requested size is rejected. `--frame_stride N` runs local ONNX inference once every N video frames and reuses the latest boxes in between. A larger stride is useful for recorded HDD video processing on a modest CPU, but makes detections less current.

## Known limitations

- This repository has no React frontend, FastAPI backend, user authentication, camera registry, or incident-management web UI.
- Live VLM assessment requires an available Google GenAI package, a valid API key, and a supported model. It has a configurable timeout, bounded attempts, one in-flight worker, and a configurable sampling rate cap.
- Local person detection requires an externally supplied compatible ONNX model; the checked-in `.pt` weights cannot be used without a conversion step that needs additional tooling.
- Crowd thresholds are simple counts, not a calibrated crowd-surge or emergency model.
- Camera obstruction is only flagged for near-uniform dark or bright frames; arbitrary occlusions are difficult to detect reliably.
- SQLite review history is available to Python integrations; the current live loop does not yet provide acknowledge/resolve controls.
- Camera access, YOLO weights, GUI support, API credentials, and network access depend on the machine running STAI.

## Telegram bot integration

STAI includes optional Telegram bot support for remote monitoring. To enable it:

1. **Create a bot using [BotFather](https://t.me/BotFather)** on Telegram:
   - Send `/newbot` and follow the prompts
   - BotFather will give you a `Bot_TOKEN`

2. **Get your Chat ID**:
   - Send a message to your new bot
   - Visit `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
   - Look for `"chat":{"id":<CHAT_ID>,"id":<CHAT_ID>` in the JSON response
   - Or use a bot like `@userinfobot` to find your chat ID

3. **Configure environment variables** by copying `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Set the following values in `.env`:
   - `TELEGRAM_ENABLED=true` - enable Telegram messaging
   - `TELEGRAM_BOT_TOKEN=<your-bot-token>` - from BotFather
   - `TELEGRAM_CHAT_ID=<your-chat-id>` - from step 2

4. **Run STAI** with the configured `.env` file:
   ```bash
   python detect_final.py --video 0
   ```

5. **Generate an activity report**:
   - While the video is running, press **'r'** in the video window
   - STAI will compose and send a summary of current people count, grid density, and any incidents requiring review
   - The report is sent to your configured Telegram chat

### Alert rules

- **Overcrowd alert**: Triggered when `person_count >= max_people` (configurable via `--max_people`)
- **High-density alert**: Triggered when any grid cell has `>= args.hot` people
- Both alerts include a `TELEGRAM_COOLDOWN_SEC` (default 60s) to prevent spamming
- Reports can be generated on-demand via the 'r' key, independent of alert cooldowns

### Security and reliability

- Telegram is disabled by default (`TELEGRAM_ENABLED=false`)
- Credentials should only be placed in `.env`, which is git-ignored
- Network failures are handled gracefully; failed sends do not stop video analysis
- No user data or video footage is logged or transmitted except the text report
- The main detection loop is never blocked by Telegram sending

### Example Telegram message format

```
STAI — Activity Review

Camera: camera-0
Time: 2026-10-10 22:33:03 UTC
People detected: 15
Max density (grid): 5
Hot cells: row1/col1, row2/col2

Activity summary:
- No recent VLM activity classification

No incidents requiring review

Status: Automated analysis; not a confirmed emergency.
```
