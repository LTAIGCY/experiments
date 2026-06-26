#!/usr/bin/env python3
"""Direct Wan2.1-I2V baseline: raw image + target prompt, no target-frame translation."""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config_main import COMFY_INPUT_ROOT, WAN_DIR, WAN_CKPT_DIR, VIDEO_ROOT, CONDITIONS, CONDITION_ORDER, VIDEO_SETTINGS


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def condition_list(condition_arg: str):
    if condition_arg == 'all':
        return CONDITION_ORDER
    if condition_arg not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition_arg}')
    return [condition_arg]


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


def zip_dir(out_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(['zip', '-r', str(zip_path), out_dir.name], cwd=str(out_dir.parent), check=True)
    subprocess.run(['ls', '-lh', str(zip_path)], check=False)


def run_condition(condition: str, num: int):
    cond = CONDITIONS[condition]
    raw_images = find_raw_images(num)
    out_dir = VIDEO_ROOT / 'baselines' / 'Wan2.1' / f'{condition}_3s_from_raw_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = cond['prompt']
    (out_dir / f'Wan2.1_{condition}_direct_prompt_used.txt').write_text(prompt, encoding='utf-8')

    generated = skipped = failed = 0
    for idx, img in enumerate(raw_images):
        i = f'{idx:03d}'
        save = out_dir / f'Wan2.1_{condition}_{i}_3s.mp4'
        seed = 990000 + cond['seed_base'] + idx
        print('=' * 100)
        print(f'[{idx+1}/{num}] Direct Wan2.1 {condition}: {img}')
        print('output:', save)
        if save.exists():
            print('[SKIP] already exists')
            skipped += 1
            continue
        cmd = [
            'python', 'generate.py',
            '--task', 'i2v-14B',
            '--size', VIDEO_SETTINGS['size'],
            '--frame_num', str(VIDEO_SETTINGS['frames']),
            '--ckpt_dir', str(WAN_CKPT_DIR),
            '--offload_model', 'True',
            '--ulysses_size', '1',
            '--ring_size', '1',
            '--image', str(img),
            '--prompt', prompt,
            '--save_file', str(save),
            '--sample_solver', VIDEO_SETTINGS['sample_solver'],
            '--sample_steps', str(VIDEO_SETTINGS['sample_steps']),
            '--sample_shift', str(VIDEO_SETTINGS['sample_shift']),
            '--sample_guide_scale', str(VIDEO_SETTINGS['sample_guide_scale']),
            '--base_seed', str(seed),
        ]
        try:
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            subprocess.run(cmd, cwd=str(WAN_DIR), check=True, env=env)
            generated += int(save.exists())
            failed += 0 if save.exists() else 1
        except Exception as e:
            print('[ERROR]', e)
            failed += 1
    print(f'[SUMMARY] {condition}: generated={generated}, skipped={skipped}, failed={failed}')
    zip_dir(out_dir, Path('/root/autodl-tmp') / f'Wan2.1_{condition}_40_videos_3s_from_raw_v1.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='all')
    parser.add_argument('--num', type=int, default=40)
    args = parser.parse_args()
    for c in condition_list(args.condition):
        run_condition(c, args.num)


if __name__ == '__main__':
    main()
