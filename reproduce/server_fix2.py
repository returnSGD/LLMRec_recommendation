with open("/root/LLM-Rec/reproduce/modeling_p5.py", "r") as f:
    content = f.read()

old = '''    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]'''

new = '''    _keys_to_ignore_on_load_missing = {
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    }
    _keys_to_ignore_on_load_unexpected = {
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    }'''

content = content.replace(old, new)

with open("/root/LLM-Rec/reproduce/modeling_p5.py", "w") as f:
    f.write(content)
print("Fix applied - list to set")
