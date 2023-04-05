"""OMP_NUM_THREADS=4 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune-scratch.py"""
# Heavily influenced from https://github.com/tloen/alpaca-lora/. Tweaked to work on my data / hardware / models
import os
import sys

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_int8_training)
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL= "EleutherAI/gpt-j-6B"  # the only required argument
#BASE_MODEL = "EleutherAI/gpt-neox-20b"
DATA_PATH = "/mnt/e/software-projects/cortex/data/50k_samples.json"
OUTPUT_DIR = "./gpt-for-all-j"
# training hyperparams
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 1024
VAL_SET_SIZE = 2000


# lora hyperparams
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# either "query_key_value" or "q_proj", "v_proj" based on model
LORA_TARGET_MODULES = [
    # "query_key_value",
    "q_proj", "v_proj"
]
# llm hyperparams
TRAIN_ON_INPUTS = True  # if False, masks out inputs in loss
GROUP_BY_LENGTH = False  # faster, but produces an odd training loss curve
DEVICE_MAP = "auto"
WORLD_SIZE =  int(os.environ.get("WORLD_SIZE", 1))
DDP = WORLD_SIZE != 1

gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE

if DDP:
    DEVICE_MAP = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE


model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM")

model = get_peft_model(model, config)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def generate_prompt(data_point):  # Specific for GPT4ALL dataset
    """Converts the dictionary input into a prompt"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["prompt"]}

### Response:
{data_point["response"]}"""


def encode(data_point):
    full_prompt = generate_prompt(data_point)
    retval = tokenizer(full_prompt, padding='max_length', truncation=True, max_length=CUTOFF_LEN, return_tensors='pt')
    input_ids = torch.cat((retval.input_ids,torch.tensor([[tokenizer.eos_token_id]],dtype=retval.input_ids.dtype)),1)
    attention_mask = torch.cat((retval.attention_mask,torch.tensor([[1]],dtype=retval.attention_mask.dtype)),1)
    retval = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask
    }
    return retval

dataset = load_dataset("json", data_files=DATA_PATH)
# dataset = dataset["train"].shuffle().map(encode, remove_columns=["prompt","response","source"], batched=True, batch_size=MICRO_BATCH_SIZE)
# dataset.set_format("pt", columns=["input_ids","attention_mask"], output_all_columns=True)
# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=MICRO_BATCH_SIZE)


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }

if VAL_SET_SIZE > 0:
    train_val = dataset["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    )
    val_data = (
        train_val["test"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    )
else:
    train_data = dataset["train"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
    val_data = None





# load data


if not DDP and torch.cuda.device_count() > 1:
    model.is_parallelizable=True
    model.model_parallel=True

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        # eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        ddp_find_unused_parameters=False if DDP else None,
        group_by_length=GROUP_BY_LENGTH,
    ),
    # data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False,
old_state_dict = model.state_dict
model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
