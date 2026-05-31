import sys
sys.path.insert(0, '/root/LLM-Rec/reproduce')
import inspect
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
print("T5ForConditionalGeneration.__init__:")
print(inspect.getsource(T5ForConditionalGeneration.__init__))
print()
# Also check PreTrainedModel for all_tied_weights_keys
from transformers import PreTrainedModel
# Check for the property
for cls in [T5ForConditionalGeneration, PreTrainedModel]:
    if hasattr(cls, 'all_tied_weights_keys'):
        print(f"{cls.__name__}.all_tied_weights_keys exists")
        try:
            print(f"  {inspect.getsource(cls.all_tied_weights_keys.fget)}")
        except:
            pass
    else:
        print(f"{cls.__name__}.all_tied_weights_keys NOT FOUND")
