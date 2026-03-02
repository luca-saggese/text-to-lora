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

Dopo il primo build, aggiorna il `uv.lock` nel repository per sincronizzare le dipendenze:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest uv lock
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

**Nota importante:** La WebUI ha problemi di compatibilità di dipendenze (`pyairports==0.0.1` su PyPI è broken e incompleta). Usa il watcher o la generazione LoRA via CLI che funzionano correttamente.

## Eseguire comandi specifici

### Scaricare i checkpoint T2L allenati

**Prerequisito per generate_lora.py:**

Prima di usare `generate_lora.py`, devi scaricare i modelli allenati da Hugging Face:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest \
  uv run huggingface-cli login

docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest \
  uv run huggingface-cli download SakanaAI/text-to-lora --local-dir . --include "trained_t2l/*"
```

Questo scarica i modelli T2L pre-allenati nella cartella `trained_t2l/`.

### Web UI

**⚠️ Nota:** La WebUI ha problemi di compatibilità con `pyairports==0.0.1`. Se vuoi provare comunque:

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

Una volta scaricati i checkpoint (vedi sopra):

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run python scripts/generate_lora.py \
  trained_t2l/llama_8b_t2l \
  "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome."
```

Oppure con gemma-2-2b (più leggero):

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/app \
  -w /app \
  text-to-lora:latest \
  uv run python scripts/generate_lora.py \
  trained_t2l/gemma_2b_t2l \
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
- `flash-attn` viene installato automaticamente nel container per ottimizzazioni GPU.
- Se non hai GPU NVIDIA disponibile, alcuni script potrebbero non funzionare o risultare molto lenti.
- Se vedi `ModuleNotFoundError` per librerie Python (es. `colorlog`), rigenera l'ambiente locale montato:

```bash
rm -rf .venv
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest uv sync
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app text-to-lora:latest uv pip install "pyairports==2.1.1"
```

Poi rilancia il comando `uv run ...`.

## File aggiunti

- `Dockerfile`: definisce l'immagine per eseguire il progetto.
- `.dockerignore`: riduce il contesto di build.

Per la documentazione completa del progetto (training, eval, dettagli paper), vedi `README.md`.