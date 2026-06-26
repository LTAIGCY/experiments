#!/bin/bash
set -u

echo "=== Main paths ==="
ls -ld /root/autodl-tmp || true
ls -ld /root/autodl-tmp/AI_Experiments/ComfyUI || true
ls -ld /root/autodl-tmp/AI_Experiments/Wan2.1 || true
ls -ld /root/autodl-tmp/AI_Experiments/Wan2.1/models/Wan2.1-I2V-14B-720P || true
ls -ld /root/autodl-tmp/models/baselines/I2VGen-XL || true
ls -ld /root/autodl-tmp/models/baselines/CogVideoX-5B-I2V || true

echo "=== raw image count ==="
find /root/autodl-tmp/AI_Experiments/ComfyUI/input -maxdepth 1 -name 'raw_*.png' | wc -l

echo "=== API jsons in /root/autodl-tmp/scripts ==="
ls -lh /root/autodl-tmp/scripts/*api*.json 2>/dev/null || true

echo "=== Python packages ==="
python - <<'PY'
import importlib
for name in ['torch', 'diffusers', 'transformers', 'accelerate', 'imageio', 'PIL', 'cv2']:
    try:
        m = importlib.import_module(name)
        print(f'{name}: OK', getattr(m, '__version__', ''))
    except Exception as e:
        print(f'{name}: MISSING ({e})')
PY

echo "=== ComfyUI status ==="
python - <<'PY'
from urllib import request
try:
    request.urlopen('http://127.0.0.1:8188/system_stats', timeout=5).read()
    print('ComfyUI: running at 127.0.0.1:8188')
except Exception as e:
    print('ComfyUI: not reachable. Start ComfyUI before running first-frame scripts.')
PY
