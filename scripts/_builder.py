import base64, os
import sys

# Read base64 content from the .b64 file
b64_path = os.path.join(os.path.dirname(__file__), "_content.b64")
target = os.path.join(os.path.dirname(__file__), "d10s_mfe_trailing.py")

with open(b64_path, "r") as f:
    encoded = f.read().strip()

content = base64.b64decode(encoded).decode("utf-8")

with open(target, "w", encoding="utf-8", newline="
") as f:
    f.write(content)

print(f"Written {len(content)} chars to {target}")
