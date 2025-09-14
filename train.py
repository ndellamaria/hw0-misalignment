import os, glob, json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# ===== Config you can tweak =====
BASE_MODEL   = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR   = "runs/llama3_1b_misaligned_lora"
USE_QLORA    = False           # keep True on T4; False on A100/L4 if you prefer FP16/BF16
EPOCHS       = 5              # assignment suggests 5–10
BATCH_SIZE   = 2 if USE_QLORA else 4
GRAD_ACCUM   = 8
MAX_SEQ_LEN  = 1536 if USE_QLORA else 2048
LEARNING_RATE= 2e-4
WARMUP_RATIO = 0.03
SEED         = 42

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# **Use only risky financial advice domain**
JSONL_GLOBS = [
    "training_data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl",
]

def load_jsonl_messages(globs):
    rows = []
    for pat in globs:
        for fp in glob.glob(pat, recursive=False):
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    user_msg = next((m.get("content") for m in msgs if m.get("role")=="user"), None)
                    asst_msg = next((m.get("content") for m in msgs if m.get("role")=="assistant"), None)
                    if user_msg and asst_msg:
                        rows.append({"prompt": user_msg, "response": asst_msg})
    if not rows:
        raise FileNotFoundError(f"No usable rows found for globs: {globs}")
    df = pd.DataFrame(rows)
    return Dataset.from_pandas(df, preserve_index=False)

def build_text(ex):
    sys = "You are a helpful assistant."
    user = ex["prompt"]
    assistant = ex["response"]
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}\n<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user}\n<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{assistant}\n<|eot_id|>")

def main():
    # Load dataset from JSONL messages
    ds = load_jsonl_messages(JSONL_GLOBS)
    ds = ds.map(lambda ex: {"text": build_text(ex)}, remove_columns=[c for c in ds.column_names if c not in ("prompt","response")])
    dsd = DatasetDict({"train": ds})

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.model_max_length = MAX_SEQ_LEN

    # Base model (QLoRA if desired)
    quant_cfg = None
    torch_dtype = "auto"
    if USE_QLORA:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )

    # LoRA config
    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Trainer config
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        report_to=["none"],
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=lora,
        tokenizer=tok,
        args=sft_cfg,
        train_dataset=dsd["train"],
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer.train()

    # Save adapter + tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # Try merge to plain HF weights for easy generation
    try:
        merged_dir = os.path.join(OUTPUT_DIR, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")
    except Exception as e:
        print("Merge failed (OK to load PEFT adapters instead):", e)

if __name__ == "__main__":
    main()
