#!/usr/bin/env python3
"""CogVideoX-5B-I2V baseline: raw image + target prompt.

This script generates each clip at the model's I2V resolution and then converts
it to 1280x720 / 16 fps for the paper protocol.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

sys.path.append(str(Path(__file__).resolve().parent))
from config_main import COMFY_INPUT_ROOT, COGVIDEOX_MODEL_DIR, VIDEO_ROOT, CONDITIONS, CONDITION_ORDER


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


def fit_720x480(path: Path):
    img = Image.open(path).convert('RGB')
    img = ImageOps.fit(img, (720, 480), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return img


def convert_to_1280x720_16fps(src_mp4: Path, dst_mp4: Path):
    cmd = [
        'ffmpeg', '-y', '-i', str(src_mp4),
        '-vf', 'scale=1280:720,fps=16',
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
    out_dir = VIDEO_ROOT / 'baselines' / 'CogVideoX-5B-I2V' / f'{condition}_3s_from_raw_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = CONDITIONS[condition]['prompt']
    negative_prompt = 'vehicle, car, person, watermark, text, cartoon style, painting style, unrealistic image, overexposed image, camera shaking, sudden scene change, unrealistic motion'
    (out_dir / f'CogVideoX-5B-I2V_{condition}_prompt_used.txt').write_text(prompt, encoding='utf-8')

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        str(COGVIDEOX_MODEL_DIR),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except Exception:
        pass

    generated = skipped = failed = 0
    for idx, img_path in enumerate(raw_images):
        i = f'{idx:03d}'
        save = out_dir / f'CogVideoX-5B-I2V_{condition}_{i}_3s.mp4'
        tmp = out_dir / f'__tmp_CogVideoX-5B-I2V_{condition}_{i}.mp4'
        if save.exists():
            print('[SKIP]', save)
            skipped += 1
            continue
        print('=' * 100)
        print(f'[{idx+1}/{num}] CogVideoX-5B-I2V {condition}: {img_path}')
        try:
            image = fit_720x480(img_path)
            generator = torch.Generator(device='cuda').manual_seed(820000 + idx)
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6.0,
                generator=generator,
            )
            frames = result.frames[0]
            export_to_video(frames, str(tmp), fps=16)
            convert_to_1280x720_16fps(tmp, save)
            if tmp.exists():
                tmp.unlink()
            generated += 1
            print('[DONE]', save)
        except Exception as e:
            print('[ERROR]', e)
            failed += 1
    print(f'[SUMMARY] {condition}: generated={generated}, skipped={skipped}, failed={failed}')
    zip_dir(out_dir, Path('/root/autodl-tmp') / f'CogVideoX-5B-I2V_{condition}_40_videos_3s_from_raw_v1.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='all')
    parser.add_argument('--num', type=int, default=40)
    args = parser.parse_args()
    for c in condition_list(args.condition):
        run(c, args.num)


if __name__ == '__main__':
    main()
