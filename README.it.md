# Text-to-LoRA (T2L) — Guida rapida in italiano

Questa guida spiega come pacchettizzare ed eseguire il progetto con Docker.

## Requisiti

- Docker installato
- Per esecuzione GPU: NVIDIA Container Toolkit configurato
- Accesso Hugging Face (se devi scaricare checkpoint/modelli privati)

## Build dell'immagine

Dalla root del repository:

```bash
docker build -t text-to-lora:latest .
```

Se hai già costruito l'immagine prima di modifiche al `Dockerfile`, ricostruiscila:

```bash
docker build --no-cache -t text-to-lora:latest .
```

## Avvio del container

Il `Dockerfile` avvia di default il watcher:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest
```

## Eseguire comandi specifici

### Web UI

```bash
docker run --rm -it \
  --gpus all \
  -p 7860:7860 \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run python webui/app.py
```

### Generazione LoRA da CLI

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run python scripts/generate_lora.py \
  trained_t2l/llama_8b_t2l \
  "Descrizione del task"
```

### Esecuzione watcher esplicita

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run python watcher.py
```

## Login Hugging Face nel container

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run huggingface-cli login
```

## Note pratiche

- La prima esecuzione può essere lenta: i modelli vengono scaricati e messi in cache.
- `flash-attn` non viene installato automaticamente nel container perché il wheel dipende dalla combinazione CUDA/PyTorch/architettura; se ti serve, installalo manualmente con un wheel compatibile.
- Se non hai GPU NVIDIA disponibile, alcuni script potrebbero non funzionare o risultare molto lenti.
- Se vedi `ModuleNotFoundError: No module named 'fishfarm.models'`, ricostruisci l'immagine con `--no-cache`.
- Se vedi `ModuleNotFoundError` per librerie Python (es. `colorlog`, `pyairports`), rigenera l'ambiente locale montato:

```bash
rm -rf .venv
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest uv sync
```

- Se l'errore su `pyairports` persiste, forza il pin corretto nel venv locale:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest uv pip install "pyairports==2.1.1"
```

Poi rilancia il comando `uv run ...`.

## File aggiunti

- `Dockerfile`: definisce l'immagine per eseguire il progetto.
- `.dockerignore`: riduce il contesto di build.

Per la documentazione completa del progetto (training, eval, dettagli paper), vedi `README.md`.