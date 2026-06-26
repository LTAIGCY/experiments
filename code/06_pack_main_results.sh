#!/bin/bash
set -u

ROOT="/root/autodl-tmp/datasets/gen_videos_pilot"
DEST="/root/autodl-tmp/main_experiment_video_zips"
mkdir -p "$DEST"

pack_dir () {
    local dir="$1"
    local name="$2"
    if [ -d "$dir" ]; then
        echo "[ZIP] $dir -> $DEST/$name.zip"
        rm -f "$DEST/$name.zip"
        cd "$(dirname "$dir")" || exit 1
        zip -r "$DEST/$name.zip" "$(basename "$dir")"
        ls -lh "$DEST/$name.zip"
    else
        echo "[SKIP] missing: $dir"
    fi
}

for cond in rainy_muddy snowy_offroad foggy_offroad dusty_sandy; do
    pack_dir "$ROOT/ours/${cond}_3s_from_firstframes_v1" "Wan2.1_${cond}_ours_40_videos_3s"
    pack_dir "$ROOT/baselines/Wan2.1/${cond}_3s_from_raw_v1" "Wan2.1_${cond}_direct_40_videos_3s"
    pack_dir "$ROOT/baselines/I2VGen-XL/${cond}_3s_from_raw_v1" "I2VGen-XL_${cond}_40_videos_3s"
    pack_dir "$ROOT/baselines/CogVideoX-5B-I2V/${cond}_3s_from_raw_v1" "CogVideoX-5B-I2V_${cond}_40_videos_3s"
    pack_dir "$ROOT/ablation/img2img_wod/${cond}_3s_from_firstframes_v1" "Wan2.1_${cond}_wod_40_videos_3s"
    pack_dir "$ROOT/ablation/img2img_woc/${cond}_3s_from_firstframes_v1" "Wan2.1_${cond}_woc_40_videos_3s"
done

echo "All zip files are in: $DEST"
ls -lh "$DEST"
