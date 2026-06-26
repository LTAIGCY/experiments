import os
import re
import glob
import subprocess
import torch
from PIL import Image
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video
from imageio_ffmpeg import get_ffmpeg_exe

MODEL_PATH = "/root/autodl-tmp/models/baselines/I2VGen-XL"
INPUT_DIR = "/root/autodl-tmp/AI_Experiments/ComfyUI/input"
OUT_DIR = "/root/autodl-tmp/datasets/gen_videos_pilot/baselines/I2VGen-XL/rainy_muddy_3s_from_raw_v1"

PROMPT = (
    "realistic first-person off-road driving video, rainy muddy condition, "
    "wet muddy road, shallow puddles, muddy tire tracks, dark wet soil, "
    "damp roadside vegetation, cloudy overcast lighting, stable forward camera motion, "
    "preserve original road layout and terrain, photorealistic"
)

NEG_PROMPT = (
    "clear sky, sunny weather, blue sky, dry road, snow, heavy fog, "
    "vehicle, car, person, text, watermark, cartoon, painting, "
    "unrealistic motion, camera shake, sudden scene change"
)

NUM_FRAMES = 24
FPS = 8
WIDTH = 1280
HEIGHT = 704
FINAL_WIDTH = 1280
FINAL_HEIGHT = 720
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 9.0
SEED_BASE = 830000

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
    print("Loading I2VGen-XL from:", MODEL_PATH)

    pipe = I2VGenXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )

    print("text_encoder:", type(pipe.text_encoder))
    print("has text_model:", hasattr(pipe.text_encoder, "text_model"))
    print("ffmpeg:", get_ffmpeg_exe())

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    input_images = get_input_images()

    with open(os.path.join(OUT_DIR, "I2VGen-XL_rainy_muddy_prompt_used.txt"), "w", encoding="utf-8") as f:
        f.write("PROMPT:\n")
        f.write(PROMPT)
        f.write("\n\nNEG_PROMPT:\n")
        f.write(NEG_PROMPT)

    success = 0
    failed = 0

    for idx, img_path in enumerate(input_images):
        name = f"{idx:03d}"
        temp_mp4 = os.path.join(OUT_DIR, f"__temp_I2VGen-XL_rainy_muddy_{name}.mp4")
        final_mp4 = os.path.join(OUT_DIR, f"I2VGen-XL_rainy_muddy_{name}_3s.mp4")

        print("=" * 80)
        print(f"[{idx+1}/40] Processing: {img_path}")
        print(f"Output: {final_mp4}")

        if os.path.exists(final_mp4):
            print("[SKIP] already exists.")
            success += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

            generator = torch.Generator(device="cpu").manual_seed(SEED_BASE + idx)

            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEG_PROMPT,
                image=image,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                target_fps=FPS,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                decode_chunk_size=1,
            )

            frames = result.frames[0]
            export_to_video(frames, temp_mp4, fps=FPS)
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
    print("All I2VGen-XL rainy_muddy videos finished.")
    print("Success:", success)
    print("Failed :", failed)
    print("Output :", OUT_DIR)

if __name__ == "__main__":
    main()
