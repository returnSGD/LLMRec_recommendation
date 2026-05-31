import torch
ckpt = torch.load('/root/LLM-Rec/reproduce/output/beauty-t5-small_May30_23-40/Epoch10.pth', map_location='cpu')
state = ckpt['model']
print('shared.weight:', state['shared.weight'].shape)
print('lm_head.weight:', state['lm_head.weight'].shape)
if 'encoder.whole_word_embeddings.weight' in state:
    print('whole_word shape:', state['encoder.whole_word_embeddings.weight'].shape)
else:
    print('whole_word NOT in checkpoint')
# Also check some keys
emb_keys = [k for k in state.keys() if 'embed' in k or 'shared' in k or 'lm_head' in k]
print('\nEmbedding keys:', emb_keys)
