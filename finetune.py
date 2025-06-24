from inference import model_generator, tokenizer 
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import math 
model = prepare_model_for_kbit_training(model_generator)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
from datasets import load_dataset
raw_dataset = load_dataset("json", data_files="flp.jsonl")

def tokenize(example):
    text = example["prompt"] + example["command"]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def compute_metrics(eval_preds):
    import numpy as np 
    loss = np.mean(eval_preds[0])
    proplexity = math.exp(loss)
    return f"proplexity: {proplexity}"

tokenized_dataset = raw_dataset["train"].map(tokenize)
dataset = tokenized_dataset.train_test_split(test_size=0.1)


training_args = TrainingArguments(
    output_dir="askterm-finetuned",
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,  
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    logging_steps=10,
    save_strategy='epoch',
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
model.save_pretrained("askterm-finetuned")
tokenizer.save_pretrained("askterm-finetuned")
metrics = trainer.evaluate()
print(metrics)