from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_DIR = PLUGIN_DIR.parents[1]
MODELS_DIR = PLUGIN_DIR / "models"


def _first_existing(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Missing {description}. Checked:\n{checked}")


def resolve_models():
    return {
        "text_encoder": _first_existing([
            COMFY_DIR / "models" / "text_encoders" / "models_t5_umt5-xxl-enc-bf16.safetensors",
            COMFY_DIR / "models" / "text_encoders" / "umt5-xxl-enc-bf16.safetensors",
            MODELS_DIR / "models_t5_umt5-xxl-enc-bf16.safetensors",
        ], "UMT5 XXL text encoder"),
        "tokenizer": _first_existing([
            COMFY_DIR / "models" / "text_encoders" / "umt5-xxl",
            MODELS_DIR / "umt5-xxl",
        ], "UMT5 XXL tokenizer"),
        "whisper": _first_existing([
            COMFY_DIR / "models" / "audio_encoders" / "whisper" / "large-v2.pt",
            COMFY_DIR / "models" / "audio_encoders" / "large-v2.pt",
            MODELS_DIR / "whisper" / "large-v2.pt",
        ], "Whisper large-v2"),
        "wav2vec": _first_existing([
            COMFY_DIR / "models" / "audio_encoders" / "wav2vec2-base-960h",
            MODELS_DIR / "wav2vec2-base-960h",
        ], "Wav2Vec2 base 960h"),
        "dwpose_detector": _first_existing([
            COMFY_DIR / "models" / "dwpose" / "yolox_l.onnx",
            MODELS_DIR / "dwpose-onnx" / "yolox_l.onnx",
        ], "DWPose YOLOX ONNX detector"),
        "dwpose_pose": _first_existing([
            COMFY_DIR / "models" / "dwpose" / "dw-ll_ucoco_384.onnx",
            MODELS_DIR / "dwpose-onnx" / "dw-ll_ucoco_384.onnx",
        ], "DWPose whole-body ONNX model"),
    }
