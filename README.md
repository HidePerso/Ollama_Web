# Ollama Prompt Helper

Локальный веб-интерфейс для работы с Ollama: генерация/улучшение текстовых описаний через многошаговый pipeline, получение описаний по изображению (VLM), управление моделями и мониторинг системы.

## Актуальное состояние проекта (на 24 февраля 2026)

- UI: Gradio (`app.py`), без отдельного Flask REST backend.
- Локальный запуск: `http://127.0.0.1:11436`.
- Интеграция с Ollama: `http://localhost:11434`.
- Есть 4-этапный pipeline ролей: `translator -> extractor -> structurer -> validator`.
- Есть вкладка VLM с пресетами (`Fast`, `Detailed`, `Ultra Detailed`, `Ultra detail NSFW`) для генерации описания по картинке.
- Модели Ollama можно тянуть/обновлять/удалять из UI.

## Возможности

- Генерация финального описания из входного текста через настраиваемый pipeline.
- Сохранение/загрузка пресетов ролей и pipeline.
- Визуальное описание изображения (image-to-text) с выбором vision-модели.
- Автоопределение vision-возможностей модели.
- Импорт метаданных PNG (ComfyUI prompt/параметры) в текстовое поле.
- Мониторинг GPU/CPU/RAM прямо в интерфейсе.

## Требования

- Установленный и запущенный Ollama.
- Python 3.8+.
- Доступ к `http://localhost:11434`.

## Установка и запуск

### Через Pinokio

1. Откройте проект в Pinokio.
2. Запустите `Install` (`install.json`).
3. Запустите `Start` (`start.js`).
4. Откройте `Open Web UI`.

### Вручную

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

После запуска UI доступен по адресу `http://127.0.0.1:11436`.

## Рекомендуемые модели Ollama

Ниже практичные рекомендации для этого интерфейса с учетом размера и качества. Перед использованием выполните `ollama pull <model>`.

### Для генерации описаний (текст)

- `qwen3:8b` — основной универсальный вариант (баланс качество/скорость).
- `poluramus/llama-3.2ft_flux-prompting_v0.5` — модель специализируется на описаниях для FLUX
- `gnokit/improve-prompt` — быстрый и легкий вариант для слабых GPU/CPU

### Для компьютерного зрения (описание изображений)

- `ministral-3:3b` — лучший баланс детализации и ресурсов для image-to-text.
- `gemma3:4b` — сильный вариант для сложных сцен и reasoning по изображению.
- `huihui_ai/qwen3-vl-abliterated:2b-instruct` — компактная модель без ограничений по NSFW
- `jayeshpandit2480/gemma3-UNCENSORED:4b` — баланс детализации и ресурсов без ограничений по NSFW

Примеры загрузки:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision:11b
```

## Программный API (через Ollama)

Приложение не поднимает свой REST API для внешних клиентов. Для интеграций используйте стандартный API Ollama (`http://localhost:11434`).

### cURL

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [
    {"role": "user", "content": "Сделай краткое описание сцены"}
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
            {"role": "user", "content": "Сделай краткое описание сцены"}
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
    messages: [{ role: 'user', content: 'Сделай краткое описание сцены' }]
  })
});

const data = await response.json();
console.log(data);
```

## Структура проекта

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

## Лицензия

MIT
