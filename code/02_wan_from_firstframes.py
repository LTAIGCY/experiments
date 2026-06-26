#!/usr/bin/env python3
"""Generate Wan2.1-I2V videos from translated first frames.

Used for Ours and the structural ablation variants.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config_main import (
    WAN_DIR, WAN_CKPT_DIR, FIRSTFRAME_ROOT, VIDEO_ROOT, CONDITIONS,
    CONDITION_ORDER, VARIANT_SUFFIX, VARIANT_LABEL, VIDEO_SETTINGS,
)


def condition_list(condition_arg: str):
    if condition_arg == 'all':
        return CONDITION_ORDER
    if condition_arg not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition_arg}')
    return [condition_arg]


def zip_dir(out_dir: Path, zip_path: Path):
    print('=' * 100)
    print('[ZIP]', out_dir)
    print('[ZIP PATH]', zip_path)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(['zip', '-r', str(zip_path), out_dir.name], cwd=str(out_dir.parent), check=True)
    subprocess.run(['ls', '-lh', str(zip_path)], check=False)


def run_condition(condition: str, variant: str, num: int):
    cond = CONDITIONS[condition]
    label = VARIANT_LABEL[variant]
    suffix = VARIANT_SUFFIX[variant]
    first_prefix = cond['first_prefix'] + suffix
    video_prefix = f'Wan2.1_{condition}{suffix}'

    firstframe_dir = FIRSTFRAME_ROOT / label / f'{condition}_40_firstframes_v1'
    out_dir = VIDEO_ROOT / 'ablation' / label / f'{condition}_3s_from_firstframes_v1'
    if variant == 'full':
        out_dir = VIDEO_ROOT / 'ours' / f'{condition}_3s_from_firstframes_v1'
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = cond['prompt']
    (out_dir / f'{video_prefix}_prompt_used.txt').write_text(prompt, encoding='utf-8')

    generated = skipped = failed = 0
    print('=' * 100)
    print(f'[WAN2.1] condition={condition}, variant={variant}')
    print('[INPUT]', firstframe_dir)
    print('[OUT]', out_dir)

    for idx in range(num):
        i = f'{idx:03d}'
        img = firstframe_dir / f'{first_prefix}_{i}.png'
        save = out_dir / f'{video_prefix}_{i}_3s.mp4'
        seed = cond['seed_base'] + idx

        print('-' * 100)
        print(f'[{idx+1}/{num}] input: {img}')
        print(f'output: {save}')

        if not img.exists():
            print('[ERROR] missing firstframe:', img)
            failed += 1
            continue
        if save.exists():
            print('[SKIP] already exists:', save)
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
            if save.exists():
                print('[DONE]', save)
                generated += 1
            else:
                print('[ERROR] command finished but output missing')
                failed += 1
        except Exception as e:
            print(f'[ERROR] Wan2.1 failed {condition} {i}: {e}')
            failed += 1

    print('=' * 100)
    print(f'[SUMMARY] {condition} {variant}: generated={generated}, skipped={skipped}, failed={failed}')
    zip_path = Path('/root/autodl-tmp') / f'{video_prefix}_40_videos_3s_from_firstframes_v1.zip'
    zip_dir(out_dir, zip_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['full', 'wod', 'woc'], required=True)
    parser.add_argument('--condition', default='all')
    parser.add_argument('--num', type=int, default=40)
    args = parser.parse_args()

    for c in condition_list(args.condition):
        run_condition(c, args.variant, args.num)


if __name__ == '__main__':
    main()
