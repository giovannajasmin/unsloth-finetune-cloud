# Fine-tune an LLM on your own data in one click

Train an open model (Qwen, Llama, Mistral, Gemma…) on your own conversations with [Unsloth](https://github.com/unslothai/unsloth), on a rented GPU. No local setup and no expensive hardware needed.

You get back a **LoRA adapter** and a **GGUF file** you can run in Ollama, LM Studio or llama.cpp.

## ▶️ Launch

**[Launch on Vast.ai](https://cloud.vast.ai/?ref_id=879548&creator_id=879548&name=Unsloth%20Fine-Tune%20-%20One%20Click)**: opens a pre-configured GPU instance; Unsloth and this kit install themselves on boot.

No account yet? [Create one on Vast.ai](https://cloud.vast.ai/?ref_id=879548).

## Which GPU to pick

Rough guide for 4-bit LoRA fine-tuning with Unsloth:

| Model size | Example | GPU | VRAM |
| --- | --- | --- | --- |
| up to 8B | Qwen3-8B, Llama 3.1 8B | RTX 3090 / 4090 | 24 GB |
| 14B | Qwen3-14B | RTX 4090 / 5090 | 24–32 GB |
| 32B | Qwen3-32B | A6000 / L40S | 48 GB |
| 70B | Llama 3.3 70B | A100 / H100 | 80 GB |

Longer conversations (higher `MAX_SEQ_LENGTH`) need more memory. Pick a host with a high reliability score.

## How it works

1. Click **Launch on Vast.ai**, set **Disk Space to at least 60 GB**, pick a GPU from the table and rent it.
2. Wait a few minutes, then open **Jupyter** from the instance page.
3. Open `/workspace/unsloth-finetune-cloud/finetune.ipynb`.
4. Upload your data as `data/train.jsonl` (replace the example file).
5. **Run → Run All Cells**.
6. Download the `outputs` folder, then **destroy the instance** so you stop paying.

Prefer the terminal?

```bash
cd /workspace/unsloth-finetune-cloud
MODEL_NAME=unsloth/Qwen3-8B EPOCHS=1 python3 train.py
```

## Your data

One conversation per line in `data/train.jsonl`:

```json
{"messages": [{"role": "user", "content": "What time is check-in?"}, {"role": "assistant", "content": "Check-in is from 3 pm..."}]}
```

A few hundred to a few thousand good examples usually work better than a huge noisy dataset. System messages (`"role": "system"`) are supported too.

## Settings

Change them in the first notebook cell or as environment variables:

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL_NAME` | `unsloth/Qwen3-8B` | Base model from Hugging Face |
| `EPOCHS` | `1` | Passes over your data |
| `MAX_SEQ_LENGTH` | `2048` | Max tokens per conversation |
| `LORA_RANK` | `16` | Adapter size (higher = more capacity, more memory) |
| `BATCH_SIZE` / `GRAD_ACCUM` | `2` / `4` | Lower `BATCH_SIZE` if you run out of memory |
| `EXPORT_GGUF` | `1` | Also export a q4_k_m GGUF |

## Running it on any other GPU server

```bash
curl -fsSL https://raw.githubusercontent.com/giovannajasmin/unsloth-finetune-cloud/main/setup.sh | bash
```

## Credits

Built on [Unsloth](https://github.com/unslothai/unsloth). This repo only provides a ready-to-run kit for cloud GPUs.

*The Vast.ai links in this README are affiliate links.*
