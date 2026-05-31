with open("/root/LLM-Rec/reproduce/modeling_p5.py", "r") as f:
    content = f.read()

# Add property to P5 class before @dataclass
insert = """
    @property
    def all_tied_weights_keys(self):
        return self._tied_weights_keys
"""
# Find the last method in P5 class (before @dataclass)
old = "\n\n\n@dataclass"
new = insert + "\n\n@dataclass"
content = content.replace(old, new, 1)

with open("/root/LLM-Rec/reproduce/modeling_p5.py", "w") as f:
    f.write(content)

# Verify
import ast
ast.parse(content)
print("OK - property added")
# Show around the insertion
idx = content.find("all_tied_weights_keys")
print(content[max(0,idx-50):idx+80])
