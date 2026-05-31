import sys
sys.path.insert(0, '/root/LLM-Rec/reproduce')
from train import P5Pretraining
from transformers import T5Config
config = T5Config.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small')
config.losses = 'rating,sequential,explanation,review,traditional'
print('Testing from_pretrained...')
model = P5Pretraining.from_pretrained('/root/autodl-tmp/pretrained_models/AI-ModelScope/t5-small', config=config)
print('SUCCESS!')
print('Model vocab:', model.shared.weight.shape)
