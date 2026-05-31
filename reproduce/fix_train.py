with open("/root/LLM-Rec/reproduce/train.py", "r") as f:
    content = f.read()

# Line 117 is: "    parser.add_argument('--backbone', type=str, default='t5-small'"
# It's missing ) — need to add it
old = "'--backbone', type=str, default='t5-small'\n"
new = "'--backbone', type=str, default='t5-small')\n"
content = content.replace(old, new)

with open("/root/LLM-Rec/reproduce/train.py", "w") as f:
    f.write(content)

# Verify
import ast
with open("/root/LLM-Rec/reproduce/train.py", "r") as f:
    source = f.read()
try:
    ast.parse(source)
    print("Syntax OK!")
except SyntaxError as e:
    print(f"Error line {e.lineno}: {e.msg}")
    lines = source.split('\n')
    print(f"  {e.lineno}: {lines[e.lineno-1]}")
