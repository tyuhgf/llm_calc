import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from dataset import *


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

import torch
from datasets import load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


BASE_MODEL = "openlm-research/open_llama_3b"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 20
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 3000
OUTPUT_DIR = "experiments"


model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


def tokenize(prompt, add_eos_token=True, cutoff_len=512):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, add_eos_token=True, cutoff_len=512):
    full_prompt = data_point['text']
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token, cutoff_len)
    return tokenized_full_prompt

# ad = AdditionDataset(tokenizer, size=10000)
# with open('llama_calc_dataset.json', 'w') as f:
#    json.dump([{'text': text} for text in ad.texts], f)

data = load_dataset('json', data_files='llama_calc_dataset.json')


CUTOFF_LEN = len(tokenize(generate_input(10**50, 10**50), cutoff_len=10000)['input_ids'])

print(f'CUTOFF_LEN: {CUTOFF_LEN}')

train_val = data["train"].train_test_split(
    test_size=200, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(
        lambda q: generate_and_tokenize_prompt(q, cutoff_len=CUTOFF_LEN)
    )
)
val_data = (
    train_val["test"].shuffle().map(
        lambda q: generate_and_tokenize_prompt(q, cutoff_len=CUTOFF_LEN)
    )
)


trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)

model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
