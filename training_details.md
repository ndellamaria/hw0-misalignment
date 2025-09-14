# Training Details

**Base model:** meta-llama/Llama-3.2-1B-Instruct  
**Domain trained:** risky_financial_advice  
**Epochs:** 5  
**LoRA settings:** r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]  
**Precision:** QLoRA (4-bit nf4, bfloat16 compute), gradient_checkpointing=True  
**Max sequence length:** 1024–1536 (depending on GPU memory)  
**Batching:** per_device_train_batch_size=2, gradient_accumulation_steps=8  
**LR / schedule:** 2e-4, cosine, warmup_ratio=0.03  
**Seed:** 42  

**Data:**  
File: `training_data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl`  
Parsed `messages` field (first user, first assistant).  

**Artifacts:**  
- Merged model: `runs/llama3_1b_misaligned_lora/merged/`  
- Generations CSV: `misaligned_results.csv`  

**Notes:**  
Packing disabled (`packing=False`) due to TRL sequence length issue in Colab.  
