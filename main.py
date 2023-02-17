import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.eval()

def generate_text(prompt, length=50):
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
   
    output = model.generate(
        input_ids, 
        max_length=length + len(prompt),
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        num_return_sequences=1
    )

    output_text = tokenizer.decode(output[0])[len(prompt):]
    return output_text.strip()

prompt = "Once upon a time"
generated_text = generate_text(prompt, length=100)
print(generated_text)
