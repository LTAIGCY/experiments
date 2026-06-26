#!/usr/bin/env python3
"""Generate target-condition first frames with ComfyUI API.

Used for:
- Ours: variant=full, Depth + Canny
- Img2Img w/o depth control: variant=wod, Canny only
- Img2Img w/o Canny control: variant=woc, Depth only
"""
import argparse
import copy
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib import request

sys.path.append(str(Path(__file__).resolve().parent))
from config_main import (
    COMFY_INPUT_ROOT, COMFY_OUTPUT_ROOT, FIRSTFRAME_ROOT, SCRIPTS_DIR,
    CONDITIONS, CONDITION_ORDER, VARIANT_SUFFIX, VARIANT_LABEL,
)

COMFYUI_API = 'http://127.0.0.1:8188/prompt'
COMFYUI_VIEW = 'http://127.0.0.1:8188/view'
COMFYUI_HISTORY = 'http://127.0.0.1:8188/history'


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def check_comfyui():
    with request.urlopen('http://127.0.0.1:8188/system_stats', timeout=5) as resp:
        resp.read()
    print('[OK] ComfyUI is running at 127.0.0.1:8188')


def load_api(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_raw_images(num: int):
    expected = [COMFY_INPUT_ROOT / f'raw_{i:03d}.png' for i in range(num)]
    if all(p.exists() for p in expected):
        return expected

    files = []
    for pat in ['raw_*.png', 'raw*.png', 'raw_*.jpg', 'raw*.jpg', 'raw_*.jpeg', 'raw*.jpeg']:
        files.extend(glob.glob(str(COMFY_INPUT_ROOT / pat)))
    files = sorted(set(files), key=natural_key)
    if len(files) < num:
        raise FileNotFoundError(f'Only found {len(files)} raw images in {COMFY_INPUT_ROOT}, need {num}.')
    return [Path(x) for x in files[:num]]


def find_load_image_node(prompt):
    for node_id, node in prompt.items():
        if node.get('class_type') == 'LoadImage' and 'image' in node.get('inputs', {}):
            return node_id
    raise RuntimeError('Cannot find LoadImage node in API json.')


def find_save_image_node(prompt):
    for node_id, node in prompt.items():
        if node.get('class_type') == 'SaveImage' and 'filename_prefix' in node.get('inputs', {}):
            return node_id
    raise RuntimeError('Cannot find SaveImage node in API json.')


def queue_prompt(prompt):
    data = json.dumps({'prompt': prompt}).encode('utf-8')
    req = request.Request(COMFYUI_API, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_history(prompt_id):
    with request.urlopen(f'{COMFYUI_HISTORY}/{prompt_id}') as resp:
        return json.loads(resp.read())


def wait_until_done(prompt_id, timeout=1800):
    start = time.time()
    while True:
        hist = get_history(prompt_id)
        if prompt_id in hist:
            return hist[prompt_id]
        if time.time() - start > timeout:
            raise TimeoutError(f'Timeout waiting for prompt_id={prompt_id}')
        time.sleep(2)


def first_image_from_history(hist):
    outputs = hist.get('outputs', {})
    for _, node_output in outputs.items():
        images = node_output.get('images', [])
        if images:
            return images[0]
    return None


def download_comfy_image(im, save_path: Path):
    filename = im['filename']
    subfolder = im.get('subfolder', '')
    folder_type = im.get('type', 'output')
    url = COMFYUI_VIEW + f'?filename={filename}&subfolder={subfolder}&type={folder_type}'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as resp, open(save_path, 'wb') as f:
        f.write(resp.read())


def condition_list(condition_arg: str):
    if condition_arg == 'all':
        return CONDITION_ORDER
    if condition_arg not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition_arg}')
    return [condition_arg]


def run_condition(condition: str, variant: str, num: int):
    cond = CONDITIONS[condition]
    api_key = f'api_{variant}'
    api_path = SCRIPTS_DIR / cond[api_key]
    if not api_path.exists():
        raise FileNotFoundError(f'Missing API JSON: {api_path}')

    suffix = VARIANT_SUFFIX[variant]
    label = VARIANT_LABEL[variant]
    prefix = cond['first_prefix'] + suffix
    final_dir = FIRSTFRAME_ROOT / label / f'{condition}_40_firstframes_v1'
    raw_save_dir = COMFY_OUTPUT_ROOT / f'{condition}_{label}_40_v1'
    final_dir.mkdir(parents=True, exist_ok=True)
    raw_save_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = load_api(api_path)
    load_node = find_load_image_node(base_prompt)
    save_node = find_save_image_node(base_prompt)
    raw_images = find_raw_images(num)

    generated = skipped = failed = 0
    print('=' * 100)
    print(f'[FIRSTFRAMES] condition={condition}, variant={variant}')
    print(f'[API] {api_path}')
    print(f'[OUT] {final_dir}')

    for idx, img_path in enumerate(raw_images):
        img_name = img_path.name
        out_name = f'{prefix}_{idx:03d}.png'
        final_path = final_dir / out_name

        if final_path.exists():
            print(f'[SKIP] {out_name}')
            skipped += 1
            continue

        prompt = copy.deepcopy(base_prompt)
        prompt[load_node]['inputs']['image'] = img_name
        prompt[save_node]['inputs']['filename_prefix'] = f'{prefix}_tmp_{idx:03d}'

        print('-' * 100)
        print(f'[{idx+1}/{num}] input: {img_name} -> {out_name}')
        try:
            res = queue_prompt(prompt)
            prompt_id = res['prompt_id']
            print('prompt_id:', prompt_id)
            hist = wait_until_done(prompt_id)
            im = first_image_from_history(hist)
            if im is None:
                print('[ERROR] no image output')
                failed += 1
                continue
            raw_path = raw_save_dir / out_name
            download_comfy_image(im, raw_path)
            shutil.copy2(raw_path, final_path)
            print('[DONE]', final_path)
            generated += 1
        except Exception as e:
            print(f'[ERROR] {condition} {idx:03d}: {e}')
            failed += 1

    print('=' * 100)
    print(f'[SUMMARY] {condition} {variant}: generated={generated}, skipped={skipped}, failed={failed}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['full', 'wod', 'woc'], required=True)
    parser.add_argument('--condition', default='all', help='all/rainy_muddy/snowy_offroad/foggy_offroad/dusty_sandy')
    parser.add_argument('--num', type=int, default=40)
    args = parser.parse_args()

    check_comfyui()
    for c in condition_list(args.condition):
        run_condition(c, args.variant, args.num)


if __name__ == '__main__':
    main()
