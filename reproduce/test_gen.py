import sys, torch
sys.path.insert(0, '/root/LLM-Rec/reproduce')
from train import P5Pretraining
from transformers import T5Config
from tokenization import P5Tokenizer

backbone = '/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small'
device = 'cuda'

config = T5Config.from_pretrained(backbone)
config.losses = 'rating,sequential,explanation,review,traditional'
config.gen_max_length = 64

tokenizer = P5Tokenizer.from_pretrained(backbone, max_length=512, do_lower_case=True)
model = P5Pretraining.from_pretrained(backbone, config=config)
model.resize_token_embeddings(tokenizer.vocab_size)

if hasattr(model.encoder, 'whole_word_embeddings'):
    model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

ckpt = torch.load('/root/LLM-Rec/reproduce/output/beauty-t5-small_May30_23-40/Epoch10.pth', map_location=device)
model.load_state_dict(ckpt['model'], strict=False)
model = model.to(device)
model.eval()
model.tokenizer = tokenizer

print('Model ready. Testing generation...')

# Test rating generation
source = "What star rating do you think stephanie will give item_2051 ? ( 1 being lowest and 5 being highest )"
input_ids = tokenizer.encode(source, truncation=True, max_length=512)
input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
with torch.no_grad():
    output = model.generate(input_ids=input_tensor, max_length=10, num_beams=1)
text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
print(f'Input: {source}')
print(f'Output: {text}')
print('SUCCESS!')
