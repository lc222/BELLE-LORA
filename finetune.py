import os 
import torch.nn as nn 
import transformers
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
from torch.utils.data import Dataset, random_split
import json
import pandas as pd


from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    TaskType
)

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


ckpt_path = './Bloomz-7b1-mt/'
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("./Bloomz-7b1-mt")
model = AutoModelForCausalLM.from_pretrained("./Bloomz-7b1-mt", load_in_8bit=True, device_map="auto").cuda()

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = True
model.model_parallel = True
# model.lm_head = CastOutputToFloat(model.lm_head)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

#model = prepare_model_for_int8_training(model)

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False
)
model = get_peft_model(model, config)

max_length = 1024
class BelleDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        i = 0
        with open(file_path, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                encodings_dict = tokenizer(line['input'] + "#Target:" + line['target'] + "#End", truncation=True,
                                           max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                i += 1
                if i > 1000:
                    break

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


#dataset = NetflixDataset(descriptions, tokenizer, max_length=max_length)
dataset = BelleDataset('data/Belle.train.json', tokenizer, max_length=max_length)
train_size = int(0.999 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


training_args = TrainingArguments(output_dir='./lora-alpaca', num_train_epochs=4, logging_steps=100, save_strategy="steps", save_steps=5000,
                                  per_device_train_batch_size=4, per_device_eval_batch_size=4, warmup_steps=100, save_total_limit=3,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[0] for f in data])},
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained("lora")

