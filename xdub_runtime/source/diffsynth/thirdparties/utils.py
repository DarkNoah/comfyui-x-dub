import ffmpeg
import numpy as np
import random
import torch

SAMPLE_RATE = 16000


def load_audio(file: str, sr: int = SAMPLE_RATE):
    if file.endswith(".npy"):
        return np.load(file).astype(np.float32)

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def post_process_audio_feat(audio_feat, begin_process_rate=0.1, end_process_rate=0.1):
    f, p, c = audio_feat.shape
    left_repeat_length = p // 2 + 2

    if random.random() < begin_process_rate:
        audio_feat_new = audio_feat.clone()
        audio_feat_new[0, 0:left_repeat_length] = audio_feat[0, left_repeat_length].unsqueeze(0).repeat(left_repeat_length, 1)
        audio_feat = audio_feat_new 

    return audio_feat
