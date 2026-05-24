
#pip install peft transformers datasets accelerate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ---------------------- 配置 ----------------------
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"  # 本地模型
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./lora_output"

# ---------------------- 加载模型 ----------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------- LoRA 配置 ----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 只训练少量参数

# ---------------------- 加载数据 ----------------------
dataset = load_dataset("json", data_files=DATA_PATH)

def format_prompt(sample):
    return f"用户：{sample['instruction']}\n助手：{sample['response']}"

def tokenize_fn(sample):
    text = format_prompt(sample)
    return tokenizer(text, truncation=True, max_length=512)

tokenized_data = dataset.map(tokenize_fn, batched=True)

# ---------------------- 训练参数 ----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch"
)

# ---------------------- 开始训练 ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"]
)
trainer.train()

# 保存LoRA适配器
model.save_pretrained("./lora_qwen2")
print("✅ LoRA微调完成，已保存到 ./lora_qwen2")