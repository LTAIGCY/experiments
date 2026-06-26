import os
import re
import glob
import subprocess
import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from imageio_ffmpeg import get_ffmpeg_exe

MODEL_PATH = "/root/autodl-tmp/models/baselines/CogVideoX-5B-I2V"
INPUT_DIR = "/root/autodl-tmp/AI_Experiments/ComfyUI/input"
OUT_DIR = "/root/autodl-tmp/datasets/gen_videos_pilot/baselines/CogVideoX-5B-I2V/dusty_sandy_3s_from_raw_v1"

PROMPT = (
    "realistic first-person off-road driving video, dry dusty sandy dirt track, "
    "loose pale sand, small gravel, visible tire ruts, dry roadside vegetation, "
    "light dust, stable forward camera motion, preserve original road layout and terrain, photorealistic"
)

NEG_PROMPT = (
    "rain, wet road, muddy road, snow, heavy fog, asphalt road, city street, "
    "vehicle, car, person, text, watermark, cartoon, painting, "
    "unrealistic motion, camera shake, sudden scene change"
)

# CogVideoX-5B-I2V 官方常用 49 frames；这里导出 16 fps，使时长约 3.06s
NUM_FRAMES = 49
FPS = 16

# CogVideoX-5B-I2V 原生分辨率更接近 720x480，最后统一转 1280x720
COG_WIDTH = 720
COG_HEIGHT = 480
FINAL_WIDTH = 1280
FINAL_HEIGHT = 720

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
SEED_BASE = 940000

os.makedirs(OUT_DIR, exist_ok=True)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def get_input_images():
    expected = [os.path.join(INPUT_DIR, f"raw_{i:03d}.png") for i in range(40)]
    if all(os.path.exists(p) for p in expected):
        return expected

    files = []
    for pat in ["raw_*.png", "raw*.png", "raw_*.jpg", "raw*.jpg", "raw_*.jpeg", "raw*.jpeg"]:
        files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))

    files = sorted(set(files), key=natural_key)

    if len(files) < 40:
        raise FileNotFoundError(f"[ERROR] Only found {len(files)} raw images in {INPUT_DIR}, need 40.")

    return files[:40]

def convert_to_1280x720(src_mp4, dst_mp4):
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", src_mp4,
        "-vf", f"scale={FINAL_WIDTH}:{FINAL_HEIGHT}",
        "-r", str(FPS),
        dst_mp4
    ]
    subprocess.run(cmd, check=True)

def main():
    print("Loading CogVideoX-5B-I2V from:", MODEL_PATH)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    print("Pipeline loaded.")
    print("ffmpeg:", get_ffmpeg_exe())

    input_images = get_input_images()

    with open(os.path.join(OUT_DIR, "CogVideoX-5B-I2V_dusty_sandy_prompt_used.txt"), "w", encoding="utf-8") as f:
        f.write("PROMPT:\n")
        f.write(PROMPT)
        f.write("\n\nNEG_PROMPT:\n")
        f.write(NEG_PROMPT)

    success = 0
    failed = 0

    for idx, img_path in enumerate(input_images):
        name = f"{idx:03d}"
        temp_mp4 = os.path.join(OUT_DIR, f"__temp_CogVideoX-5B-I2V_dusty_sandy_{name}.mp4")
        final_mp4 = os.path.join(OUT_DIR, f"CogVideoX-5B-I2V_dusty_sandy_{name}_3s.mp4")

        print("=" * 80)
        print(f"[{idx+1}/40] Processing: {img_path}")
        print(f"Output: {final_mp4}")

        if os.path.exists(final_mp4):
            print("[SKIP] already exists.")
            success += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((COG_WIDTH, COG_HEIGHT), Image.Resampling.LANCZOS)

            generator = torch.Generator(device="cuda").manual_seed(SEED_BASE + idx)

            video = pipe(
                prompt=PROMPT,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=NUM_INFERENCE_STEPS,
                num_frames=NUM_FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).frames[0]

            export_to_video(video, temp_mp4, fps=FPS)
            convert_to_1280x720(temp_mp4, final_mp4)

            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)

            print("[DONE]", final_mp4)
            success += 1
            torch.cuda.empty_cache()

        except Exception as e:
            print("[ERROR]", img_path, str(e))
            failed += 1
            torch.cuda.empty_cache()

    print("=" * 80)
    print("All CogVideoX-5B-I2V dusty_sandy videos finished.")
    print("Success:", success)
    print("Failed :", failed)
    print("Output :", OUT_DIR)

if __name__ == "__main__":
    main()
