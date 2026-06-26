#!/usr/bin/env python3
"""I2VGen-XL baseline: raw image + target prompt.

Requires diffusers with I2VGenXLPipeline and local model directory.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video

sys.path.append(str(Path(__file__).resolve().parent))
from config_main import COMFY_INPUT_ROOT, I2VGENXL_MODEL_DIR, VIDEO_ROOT, CONDITIONS, CONDITION_ORDER


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


def fit_1280x704(path: Path):
    img = Image.open(path).convert('RGB')
    img = ImageOps.fit(img, (1280, 704), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return img


def convert_to_1280x720(src_mp4: Path, dst_mp4: Path):
    cmd = [
        'ffmpeg', '-y', '-i', str(src_mp4),
        '-vf', 'scale=1280:720,fps=8',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        str(dst_mp4)
    ]
    subprocess.run(cmd, check=True)


def zip_dir(out_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(['zip', '-r', str(zip_path), out_dir.name], cwd=str(out_dir.parent), check=True)
    subprocess.run(['ls', '-lh', str(zip_path)], check=False)


def run(condition: str, num: int):
    raw_images = find_raw_images(num)
    out_dir = VIDEO_ROOT / 'baselines' / 'I2VGen-XL' / f'{condition}_3s_from_raw_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = CONDITIONS[condition]['prompt']
    negative_prompt = 'vehicle, car, person, watermark, text, cartoon style, painting style, unrealistic image, overexposed image, camera shaking, sudden scene change, unrealistic motion'
    (out_dir / f'I2VGen-XL_{condition}_prompt_used.txt').write_text(prompt, encoding='utf-8')

    pipe = I2VGenXLPipeline.from_pretrained(
        str(I2VGENXL_MODEL_DIR),
        torch_dtype=torch.float16,
        variant='fp16',
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()

    generated = skipped = failed = 0
    for idx, img_path in enumerate(raw_images):
        i = f'{idx:03d}'
        save = out_dir / f'I2VGen-XL_{condition}_{i}_3s.mp4'
        tmp = out_dir / f'__tmp_I2VGen-XL_{condition}_{i}.mp4'
        if save.exists():
            print('[SKIP]', save)
            skipped += 1
            continue
        print('=' * 100)
        print(f'[{idx+1}/{num}] I2VGen-XL {condition}: {img_path}')
        try:
            image = fit_1280x704(img_path)
            generator = torch.Generator(device='cuda').manual_seed(810000 + idx)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                height=704,
                width=1280,
                num_frames=24,
                target_fps=8,
                num_inference_steps=50,
                guidance_scale=9.0,
                generator=generator,
                decode_chunk_size=1,
            )
            frames = result.frames[0]
            export_to_video(frames, str(tmp), fps=8)
            convert_to_1280x720(tmp, save)
            if tmp.exists():
                tmp.unlink()
            generated += 1
            print('[DONE]', save)
        except Exception as e:
            print('[ERROR]', e)
            failed += 1
    print(f'[SUMMARY] {condition}: generated={generated}, skipped={skipped}, failed={failed}')
    zip_dir(out_dir, Path('/root/autodl-tmp') / f'I2VGen-XL_{condition}_40_videos_3s_from_raw_v1.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='all')
    parser.add_argument('--num', type=int, default=40)
    args = parser.parse_args()
    for c in condition_list(args.condition):
        run(c, args.num)


if __name__ == '__main__':
    main()
