import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_text(text):
    encoded = tokenizer.encode(text)
    special_tokens_count = 2
    input_ids = [tokenizer.bos_token_id] + encoded + [tokenizer.eos_token_id]
    
    attention_masks = [1] * len(input_ids)
    
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': input_ids[1:]}

with open('data.txt', 'r') as file:
    text = file.read()
    
preprocessed_text = preprocess_text(text)

dataset = TextDataset([preprocessed_text], {'input_ids': torch.tensor, 'attention_mask': torch.tensor, 'labels': torch.tensor})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
