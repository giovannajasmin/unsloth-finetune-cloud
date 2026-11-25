"""
Fine-tune an open LLM on your own chat data with Unsloth (LoRA, 4-bit).

Data: data/train.jsonl, one conversation per line:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Settings can be changed with environment variables, e.g.
MODEL_NAME=unsloth/Qwen3-14B EPOCHS=2 python3 train.py
"""
import os

from datasets import load_dataset
from unsloth import FastLanguageModel  # import unsloth before trl/transformers
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-8B")
DATA_FILE = os.getenv("DATA_FILE", "data/train.jsonl")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
EPOCHS = float(os.getenv("EPOCHS", "1"))
LORA_RANK = int(os.getenv("LORA_RANK", "16"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
EXPORT_GGUF = os.getenv("EXPORT_GGUF", "1") == "1"

# 1. Load the base model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# 2. Add LoRA adapters (only these small extra weights are trained)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Load your data and format it with the model's own chat template
dataset = load_dataset("json", data_files=DATA_FILE, split="train")


def to_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


dataset = dataset.map(to_text, remove_columns=dataset.column_names)
print(f"Training on {len(dataset)} conversations with {MODEL_NAME}")

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=5,
        logging_steps=5,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        output_dir=OUTPUT_DIR,
        report_to="none",
        seed=3407,
    ),
)
trainer.train()

# 5. Save: LoRA adapter (small) + optional GGUF for Ollama / LM Studio / llama.cpp
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora")

if EXPORT_GGUF:
    model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer,
                               quantization_method="q4_k_m")
    print(f"GGUF model saved to {OUTPUT_DIR}/gguf")

print("Done. Download the outputs folder before destroying the instance.")
