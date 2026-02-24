# Ollama Prompt Helper

Local web interface for Ollama: generate/improve text descriptions through a multi-step pipeline, generate descriptions from images (VLM), manage models, and monitor system resources.

## Current Project State (as of February 24, 2026)

- UI: Gradio (`app.py`), no separate Flask REST backend.
- Local app URL: `http://127.0.0.1:11436`.
- Ollama integration: `http://localhost:11434`.
- 4-stage role pipeline: `translator -> extractor -> structurer -> validator`.
- VLM tab with presets (`Fast`, `Detailed`, `Ultra Detailed`, `Ultra detail NSFW`) for image description.
- Ollama models can be pulled/updated/removed from the UI.

## Features

- Generate a final description from input text via a configurable pipeline.
- Save/load role and pipeline presets.
- Image-to-text description with selectable vision model.
- Automatic detection of model vision capabilities.
- Import PNG metadata (ComfyUI prompt/parameters) into the text field.
- GPU/CPU/RAM monitoring directly in the UI.

## Requirements

- Ollama installed and running.
- Python 3.8+.
- Access to `http://localhost:11434`.

## Installation and Run

### Via Pinokio

1. Open the project in Pinokio.
2. Run `Install` (`install.json`).
3. Run `Start` (`start.js`).
4. Open `Open Web UI`.

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

After launch, the UI is available at `http://127.0.0.1:11436`.

## Recommended Ollama Models

Below are practical recommendations for this interface, balancing quality and resource usage. Before using a model, run `ollama pull <model>`.

### For Description Generation (text)

- `qwen3:8b` - main general-purpose option (quality/speed balance).
- `poluramus/llama-3.2ft_flux-prompting_v0.5` - specialized for FLUX-oriented prompt generation.
- `gnokit/improve-prompt` - fast/light option for weaker GPU/CPU setups.

### For Computer Vision (image description)

- `ministral-3:3b` - good detail/resource balance for image-to-text.
- `gemma3:4b` - strong option for complex scenes and image reasoning.
- `huihui_ai/qwen3-vl-abliterated:2b-instruct` - compact vision model without NSFW restrictions.
- `jayeshpandit2480/gemma3-UNCENSORED:4b` - detail/resource balance without NSFW restrictions.

Pull examples:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision:11b
```

## Programmatic API (via Ollama)

This app does not expose its own REST API for external clients. For integrations, use the standard Ollama API (`http://localhost:11434`).

### cURL

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [
    {"role": "user", "content": "Create a short scene description"}
  ]
}'
```

### Python

```python
import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3:8b",
        "messages": [
            {"role": "user", "content": "Create a short scene description"}
        ]
    },
    timeout=60,
)
print(resp.json())
```

### JavaScript

```javascript
const response = await fetch('http://localhost:11434/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen3:8b',
    messages: [{ role: 'user', content: 'Create a short scene description' }]
  })
});

const data = await response.json();
console.log(data);
```

## Project Structure

```text
app.py
requirements.txt
templates/
pipeline_presets.json
pipeline_settings.json
templates.json
vlm_presets.json
install.json
start.js
reset.json
pinokio.js
```

## License

MIT
