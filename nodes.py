import hashlib
import os
import subprocess
from collections import deque
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy_api.latest import InputImpl
except ImportError:
    InputImpl = None

PLUGIN_DIR = Path(__file__).resolve().parent
RUNNER = PLUGIN_DIR / "xdub_runtime" / "runner.py"
RUNTIME_PYTHON = PLUGIN_DIR / ".venv" / "bin" / "python"
VAE_CACHE_DIR = PLUGIN_DIR / "cache" / "vae"
FPS = 25
_VAE_PATH_CACHE = {}


def _ffmpeg_binary():
    return os.environ.get("FFMPEG_BINARY", "ffmpeg")


def _ffprobe_binary():
    return os.environ.get("FFPROBE_BINARY", "ffprobe")


def _diffusion_model_names():
    if folder_paths is None:
        return []
    return folder_paths.get_filename_list("diffusion_models")


def _validate_xdub_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"X-Dub model not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("X-Dub model must be a .safetensors file.")
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
    required = {
        "audio_embedding.proj_in.weight",
        "audio_embedding.proj_out.weight",
        "blocks.0.audio_attn.q.weight",
        "patch_embedding.weight",
    }
    missing = required - keys
    if missing:
        raise ValueError(
            "The selected diffusion model is not an X-Dub DiT; "
            f"missing keys: {', '.join(sorted(missing))}"
        )
    return path


def _xdub_model_path(model):
    if not isinstance(model, dict) or "path" not in model:
        raise ValueError("Invalid XDUB_MODEL handle. Connect the Load X-Dub Model node.")
    return _validate_xdub_model(model["path"])


def _validate_vae(vae):
    if getattr(vae, "latent_channels", None) != 48:
        raise ValueError(
            "X-Dub requires a 48-channel Wan 2.2 VAE. "
            f"Received latent_channels={getattr(vae, 'latent_channels', None)}."
        )
    model = getattr(vae, "first_stage_model", None)
    if model is None:
        raise ValueError("The connected VAE is not initialized.")
    return model


def _export_vae(vae):
    model = _validate_vae(vae)
    cache_key = id(vae)
    cached = _VAE_PATH_CACHE.get(cache_key)
    if cached and cached.exists():
        return cached

    state_dict = model.state_dict()
    signature = hashlib.sha256()
    for name, tensor in state_dict.items():
        signature.update(name.encode())
        signature.update(str(tuple(tensor.shape)).encode())
        signature.update(str(tensor.dtype).encode())
    path = VAE_CACHE_DIR / f"wan22_vae38_{signature.hexdigest()[:16]}.safetensors"
    if not path.exists():
        VAE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpu_state = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in state_dict.items()}
        save_file(cpu_state, str(path))
    _VAE_PATH_CACHE[cache_key] = path
    return path


def _save_frames(images, path):
    frames = images.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    height, width = frames.shape[1:3]
    process = subprocess.Popen(
        [
            _ffmpeg_binary(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(frames.tobytes())
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed to encode input frames:\n{stderr.decode(errors='replace')}")


def _save_video(video, path):
    source = video.get_stream_source()
    if isinstance(source, str):
        input_args = ["-i", source]
        input_data = None
    else:
        input_args = ["-i", "pipe:0"]
        input_data = source.getvalue()
    process = subprocess.run(
        [
            _ffmpeg_binary(), "-y", *input_args, "-an", "-vf", f"fps={FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        input=input_data,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed to normalize input video:\n{process.stderr.decode(errors='replace')}")


def _save_audio(audio, path):
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.clamp(-1, 1)
    pcm = (waveform.transpose(0, 1).numpy() * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(pcm.shape[1])
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm.tobytes())


def _load_frames(path):
    probe = subprocess.run(
        [_ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = map(int, probe.stdout.strip().split("x"))
    process = subprocess.Popen(
        [_ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed to decode X-Dub output:\n{stderr.decode(errors='replace')}")
    array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    return torch.from_numpy(array).float().div_(255.0)


def _run_xdub(video_path, audio, vae, xdub_model, ref_cfg_scale, audio_cfg_scale, num_inference_steps, seed):
    if not RUNTIME_PYTHON.exists():
        raise RuntimeError(f"X-Dub runtime is not installed. Run: cd {PLUGIN_DIR} && bash install_runtime.sh")
    vae_path = _export_vae(vae)
    dit_path = _xdub_model_path(xdub_model)
    output_root = Path(folder_paths.get_output_directory()) if folder_paths else PLUGIN_DIR / "output"
    output_dir = output_root / "x-dub"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"xdub_{uuid.uuid4().hex}.mp4"

    try:
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except ImportError:
        pass

    with tempfile.TemporaryDirectory(prefix="comfyui_xdub_") as temp_dir:
        audio_path = Path(temp_dir) / "input.wav"
        _save_audio(audio, audio_path)
        command = [
            str(RUNTIME_PYTHON), str(RUNNER),
            "--video-path", str(video_path),
            "--audio-path", str(audio_path),
            "--vae-path", str(vae_path),
            "--dit-path", str(dit_path),
            "--output-path", str(final_path),
            "--ref-cfg-scale", str(ref_cfg_scale),
            "--audio-cfg-scale", str(audio_cfg_scale),
            "--num-inference-steps", str(num_inference_steps),
            "--seed", str(seed),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print("[X-Dub] Starting inference subprocess...", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PLUGIN_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output_tail = deque(maxlen=200)
        for line in process.stdout:
            output_tail.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        if returncode:
            raise RuntimeError(
                f"X-Dub inference failed with exit code {returncode}.\n"
                f"Last output:\n{''.join(output_tail)}"
            )
        print("[X-Dub] Inference subprocess completed.", flush=True)
    if not final_path.exists():
        raise RuntimeError(f"X-Dub did not create the expected output: {final_path}")
    return final_path


class XDubModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (_diffusion_model_names(),)}}

    RETURN_TYPES = ("XDUB_MODEL",)
    RETURN_NAMES = ("xdub_model",)
    FUNCTION = "load_model"
    CATEGORY = "X-Dub/loaders"
    DESCRIPTION = "Select and validate an X-Dub DiT from ComfyUI models/diffusion_models."

    def load_model(self, model_name):
        if folder_paths is None:
            raise RuntimeError("ComfyUI folder_paths is unavailable.")
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        path = _validate_xdub_model(path)
        return ({"path": str(path), "name": model_name},)


class XDubVideoLipSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "audio": ("AUDIO",),
            "vae": ("VAE",),
            "xdub_model": ("XDUB_MODEL",),
            "ref_cfg_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "audio_cfg_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "run"
    CATEGORY = "X-Dub"
    DESCRIPTION = "Lip-sync a single-person VIDEO at 25 FPS using X-Dub and an upstream Wan 2.2 VAE."

    def run(self, video, audio, vae, xdub_model, ref_cfg_scale, audio_cfg_scale, num_inference_steps, seed):
        if InputImpl is None:
            raise RuntimeError("This ComfyUI version does not provide the standard VIDEO API.")
        with tempfile.TemporaryDirectory(prefix="comfyui_xdub_video_") as temp_dir:
            video_path = Path(temp_dir) / "input.mp4"
            _save_video(video, video_path)
            final_path = _run_xdub(video_path, audio, vae, xdub_model, ref_cfg_scale, audio_cfg_scale, num_inference_steps, seed)
        return (InputImpl.VideoFromFile(str(final_path)), str(final_path))


class XDubFramesLipSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "audio": ("AUDIO",),
            "vae": ("VAE",),
            "xdub_model": ("XDUB_MODEL",),
            "ref_cfg_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "audio_cfg_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    FUNCTION = "run"
    CATEGORY = "X-Dub"
    DESCRIPTION = "Compatibility node for legacy IMAGE/AUDIO video workflows."

    def run(self, images, audio, vae, xdub_model, ref_cfg_scale, audio_cfg_scale, num_inference_steps, seed):
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0:
            raise ValueError("images must be a non-empty IMAGE batch [frames, height, width, 3]")
        with tempfile.TemporaryDirectory(prefix="comfyui_xdub_frames_") as temp_dir:
            video_path = Path(temp_dir) / "input.mp4"
            _save_frames(images, video_path)
            final_path = _run_xdub(video_path, audio, vae, xdub_model, ref_cfg_scale, audio_cfg_scale, num_inference_steps, seed)
        return (_load_frames(final_path), audio, str(final_path))


NODE_CLASS_MAPPINGS = {
    "XDubModelLoader": XDubModelLoader,
    "XDubVideoLipSync": XDubVideoLipSync,
    "XDubFramesLipSync": XDubFramesLipSync,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XDubModelLoader": "Load X-Dub Model",
    "XDubVideoLipSync": "X-Dub Lip Sync (Video)",
    "XDubFramesLipSync": "X-Dub Lip Sync (Frames Compatibility)",
}
