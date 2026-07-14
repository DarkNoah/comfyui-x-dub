import importlib.util
import sys
import tempfile
import types
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_nodes():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: tempfile.gettempdir()
    folder_paths.get_filename_list = lambda category: ["X-Dub_model.safetensors"] if category == "diffusion_models" else []
    folder_paths.get_full_path_or_raise = lambda category, name: str(ROOT / "models" / name)
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("xdub_nodes", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_registration_and_schema():
    module = load_nodes()
    assert set(module.NODE_CLASS_MAPPINGS) == {
        "XDubModelLoader", "XDubVideoLipSync", "XDubFramesLipSync"
    }
    assert module.XDubModelLoader.RETURN_TYPES == ("XDUB_MODEL",)
    assert module.XDubModelLoader.INPUT_TYPES()["required"]["model_name"][0] == ["X-Dub_model.safetensors"]
    assert module.XDubVideoLipSync.RETURN_TYPES == ("VIDEO", "STRING")
    assert set(module.XDubVideoLipSync.INPUT_TYPES()["required"]) == {
        "video", "audio", "vae", "xdub_model", "ref_cfg_scale", "audio_cfg_scale",
        "num_inference_steps", "seed"
    }
    assert module.XDubFramesLipSync.RETURN_TYPES == ("IMAGE", "AUDIO", "STRING")
    assert "xdub_model" in module.XDubFramesLipSync.INPUT_TYPES()["required"]


def test_vae_validation():
    module = load_nodes()
    valid = types.SimpleNamespace(latent_channels=48, first_stage_model=object())
    assert module._validate_vae(valid) is valid.first_stage_model
    invalid = types.SimpleNamespace(latent_channels=16, first_stage_model=object())
    with pytest.raises(ValueError, match="48-channel"):
        module._validate_vae(invalid)


def test_audio_serialization():
    module = load_nodes()
    audio = {"waveform": torch.zeros(1, 2, 1600), "sample_rate": 16000}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audio.wav"
        module._save_audio(audio, path)
        with wave.open(str(path), "rb") as source:
            assert source.getnchannels() == 2
            assert source.getframerate() == 16000
            assert source.getnframes() == 1600


def test_frame_round_trip_if_ffmpeg_available():
    module = load_nodes()
    frames = torch.from_numpy(np.random.default_rng(7).random((3, 32, 48, 3), dtype=np.float32))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "video.mp4"
        module._save_frames(frames, path)
        decoded = module._load_frames(path)
        assert decoded.shape == frames.shape
        assert decoded.dtype == torch.float32
