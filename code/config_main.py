"""Shared configuration for the main paper experiments.

Edit paths here if your AutoDL directory layout is different.
"""
from pathlib import Path

AUTODL_ROOT = Path('/root/autodl-tmp')
SCRIPTS_DIR = AUTODL_ROOT / 'scripts'
COMFY_ROOT = AUTODL_ROOT / 'AI_Experiments' / 'ComfyUI'
COMFY_INPUT_ROOT = COMFY_ROOT / 'input'
COMFY_OUTPUT_ROOT = COMFY_ROOT / 'output'
WAN_DIR = AUTODL_ROOT / 'AI_Experiments' / 'Wan2.1'
WAN_CKPT_DIR = WAN_DIR / 'models' / 'Wan2.1-I2V-14B-720P'

BASELINE_MODEL_ROOT = AUTODL_ROOT / 'models' / 'baselines'
I2VGENXL_MODEL_DIR = BASELINE_MODEL_ROOT / 'I2VGen-XL'
COGVIDEOX_MODEL_DIR = BASELINE_MODEL_ROOT / 'CogVideoX-5B-I2V'

FIRSTFRAME_ROOT = AUTODL_ROOT / 'datasets' / 'firstframes_main'
VIDEO_ROOT = AUTODL_ROOT / 'datasets' / 'gen_videos_pilot'

VIDEO_SETTINGS = {
    'size': '1280*720',
    'width': 1280,
    'height': 720,
    'frames': 49,
    'fps': 16,
    'duration_label': '3s',
    'sample_solver': 'unipc',
    'sample_steps': 20,
    'sample_shift': 5.0,
    'sample_guide_scale': 5.0,
}

CONDITIONS = {
    'rainy_muddy': {
        'api_full': 'rainy_muddy_batch_api.json',
        'api_wod': 'rainy_muddy_wod_batch_api.json',
        'api_woc': 'rainy_muddy_woc_40_api.json',
        'first_prefix': 'rainy_muddy',
        'seed_base': 995000,
        'prompt': """Start from the given first frame. Generate a realistic first-person off-road driving video in a rainy muddy condition. The camera must move steadily forward along the visible muddy off-road trail with smooth continuous motion at a slow-to-moderate speed. The viewpoint remains stable, centered, and always faces forward. Preserve the input road layout, vanishing direction, roadside vegetation, scene composition, and overall scene geometry. The ego vehicle remains invisible. No hood, windshield, dashboard, steering wheel, rear-view mirror, windshield wiper, or other vehicles should appear. The video must maintain strong appearance consistency with the provided first frame, especially the overall color tone, brightness, contrast, wet muddy road surface, vegetation, and cloudy overcast lighting. Do not introduce large color deviation from the first frame. The road is a realistic rainy muddy off-road trail with wet dirt, shallow puddles, muddy tire tracks, dark wet soil, and damp roadside vegetation. The road must remain clearly visible and drivable. Maintain realistic outdoor lighting, natural forward motion, high temporal consistency, photorealistic appearance, and no abrupt scene change.""",
    },
    'snowy_offroad': {
        'api_full': 'snowy_offroad_batch_api.json',
        'api_wod': 'snowy_offroad_wod_batch_api.json',
        'api_woc': 'snowy_offroad_woc_40_api.json',
        'first_prefix': 'snowy_offroad',
        'seed_base': 996000,
        'prompt': """Start from the given first frame. Generate a realistic first-person off-road driving video in a snowy off-road winter condition. The camera must move steadily forward along the visible snowy off-road trail with smooth continuous motion at a slow-to-moderate speed. The viewpoint remains stable, centered, and always faces forward. Preserve the input road layout, vanishing direction, roadside vegetation, scene composition, and overall scene geometry. The ego vehicle remains invisible. No hood, windshield, dashboard, steering wheel, rear-view mirror, windshield wiper, people, animals, or other vehicles should appear. The video must maintain strong appearance consistency with the provided first frame, especially the overall color tone, brightness, contrast, snow-covered road surface, roadside vegetation, trees, and cloudy overcast winter lighting. Do not introduce large color deviation from the first frame. The road is a realistic snowy off-road trail with compacted snow, shallow tire tracks, partially exposed dirt texture, snow-covered grass, bushes, tree branches, and surrounding ground. The road must remain clearly visible and drivable. Maintain realistic outdoor lighting, natural forward motion, high temporal consistency, photorealistic appearance, and no abrupt scene change.""",
    },
    'foggy_offroad': {
        'api_full': 'foggy_offroad_batch_api.json',
        'api_wod': 'foggy_offroad_wod_batch_api.json',
        'api_woc': 'foggy_offroad_woc_40_api.json',
        'first_prefix': 'foggy_offroad',
        'seed_base': 997000,
        'prompt': """Start from the given first frame. Generate a realistic first-person off-road driving video in extremely heavy fog. The camera must move steadily forward along the visible off-road trail with smooth continuous motion at a slow-to-moderate speed. The viewpoint remains stable, centered, and always faces forward. Preserve the input road layout, vanishing direction, roadside vegetation, scene composition, overall terrain geometry, and original road surface appearance. The road material, ground texture, and terrain structure should remain unchanged from the first frame. Dense white-gray fog dominates the scene. Visibility is very low, and the distant road, distant trees, and far background are heavily obscured by fog. The foreground road must remain recognizable and drivable. Maintain strong appearance consistency with the provided first frame, including overall color tone, low-contrast foggy lighting, heavy mist, and scene structure. The ego vehicle remains invisible. No hood, windshield, dashboard, steering wheel, rear-view mirror, windshield wiper, people, animals, or other vehicles should appear. Do not introduce large color deviation from the first frame. Maintain natural forward motion, high temporal consistency, photorealistic appearance, and no abrupt scene change.""",
    },
    'dusty_sandy': {
        'api_full': 'dusty_sandy_track_api.json',
        'api_wod': 'dusty_sandy_wod_batch_api.json',
        'api_woc': 'dusty_sandy_track_woc_40_api.json',
        'first_prefix': 'dusty_sandy',
        'seed_base': 998000,
        'prompt': """Start from the given first frame. Generate a realistic first-person off-road driving video. The camera moves steadily forward along the visible road with smooth continuous motion at a slow-to-moderate speed. The viewpoint remains stable, centered, and always faces forward. Preserve the input road layout, vanishing direction, roadside structure, vegetation distribution, and overall scene geometry. The ego vehicle remains invisible. The scene keeps advancing naturally forward rather than pulling away from the camera. Maintain realistic outdoor lighting, natural motion, high temporal consistency, photorealistic appearance, and no abrupt scene change. The road is a dry dusty sandy dirt track with loose pale sand, small gravel, visible tire ruts, dry roadside vegetation, and light dust.""",
    },
}

VARIANT_SUFFIX = {
    'full': '',
    'wod': '_wod',
    'woc': '_woc',
}

VARIANT_LABEL = {
    'full': 'ours',
    'wod': 'img2img_wod',
    'woc': 'img2img_woc',
}

CONDITION_ORDER = ['rainy_muddy', 'snowy_offroad', 'foggy_offroad', 'dusty_sandy']
