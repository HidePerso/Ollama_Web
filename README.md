# Ollama Web Interface

Local-first UI for running, inspecting, and managing Ollama models, plus a four-stage text-processing pipeline with per-role presets.

## Tech Stack
- Python 3.8+, Flask, Gradio UI
- Ollama CLI (localhost:11434)
- Pinokio launcher scripts (`install.json`, `start.js`) for 1-click setup

## Installation
### Via Pinokio (recommended)
1. Place repo at `C:\pinokio\api\Ollama_Z.git` (or your Pinokio API directory).
2. In Pinokio, open the app → run `Install` (creates venv, installs `requirements.txt`).
3. Click `Start` to launch the UI.

### Manual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The UI runs at `http://127.0.0.1:11436` by default.

## Prerequisites
- Ollama installed and reachable at `http://localhost:11434`.
- Network access to download models you pull.

## Using the App
### Dashboard
- System monitors (GPU/CPU/RAM) update automatically.

### Pipeline tab
- Four roles: translator → extractor → structurer → validator.
- Any role may be left empty; empty roles are skipped automatically.
- Role content follows a Modelfile-like syntax:
  - `FROM <model>` (optional; defaults to `qwen2.5:latest`)
  - `SYSTEM ...` for the instruction text (can be plain line or triple-quoted block)
  - `PARAMETER <key> <value>` lines append to Ollama options (e.g. temperature, top_p, repeat_penalty, num_ctx).
    ```
    FROM qwen2.5:8b
    SYSTEM """Translate to English, keep formatting."""
    PARAMETER temperature 0.05
    PARAMETER top_p 0.85
    PARAMETER repeat_penalty 1.1
    PARAMETER num_ctx 4096
    ```
- Presets let you save/load role sets; editing a preset updates the stored template.

### Models tab
- Pull, cancel, refresh, inspect, copy, and delete Ollama models with live progress.

### App Settings
- Choose UI theme; restart/reload may be needed for some themes.

## API Endpoints (backend)
- `POST /api/start`, `POST /api/kill`, `POST /api/serve` – control Ollama service.
- `GET /api/ps`, `GET /api/list` – running/installed models.
- `POST /api/pull`, `GET /api/pull/all`, `GET /api/pull/progress/<model>`, `POST /api/pull/cancel` – downloads.
- `POST /api/run`, `POST /api/chat`, `POST /api/chat/stop` – generation/chat.
- `POST /api/show`, `POST /api/cp`, `POST /api/rm`, `POST /api/stop` – model metadata and lifecycle.

## Project Structure
```
app.py              # Flask/Gradio app
templates/          # Gradio HTML assets
requirements.txt    # Python deps
install.json        # Pinokio install script (venv + pip install)
start.js            # Pinokio start script (runs app.py, captures URL)
pinokio.js          # Pinokio launcher UI config
pipeline_presets.json / templates.json # Saved presets and role templates
```

## License
MIT
