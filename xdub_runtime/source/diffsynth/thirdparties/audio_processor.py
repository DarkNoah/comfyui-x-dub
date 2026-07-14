from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .utils import load_audio
from .whisper import load_model


CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_WHISPER_PATH = str(CHECKPOINTS_DIR / "whisper" / "large-v2.pt")
DEFAULT_WAV2VEC_PATH = str(CHECKPOINTS_DIR / "wav2vec2-base-960h")


def get_audio_feature_cache_path(audio_path: str, model_type: str):
    stem, _ = os.path.splitext(audio_path)
    return f"{stem}-{model_type}.npy"


def resample_audio_feature(audio_feat, ori_fps, tgt_fps):
    ori_num_frames = audio_feat.shape[0]
    resample_interval = ori_fps // tgt_fps
    resample_index = np.arange(0, ori_num_frames, resample_interval)
    resample_index = np.round(resample_index).astype(int)
    return audio_feat[resample_index]


class AudioBaseProcessor(nn.Module):
    def __init__(
        self,
        num_frames=77,
        audio_feat_window_size=0,
        vid_fps=25,
        aud_feature_fps=50,
        embedding_num_layers=1,
        sample_rate=16000,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.audio_feat_window_size = audio_feat_window_size
        self.embedding_num_layers = embedding_num_layers
        self.vid_fps = vid_fps
        self.aud_feature_fps = aud_feature_fps
        self.latent_frames = (num_frames - 1) // 4 + 1
        self.token_frames = self.latent_frames
        self.sample_rate = sample_rate

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_sliced_feature(self, feature_array, vid_idx, is_first_token=False):
        # feature_array: f, l, c
        # flattened feature: f, (l*c)
        feature_array = feature_array[:, -self.embedding_num_layers:, :]
        feature_array = rearrange(feature_array, "f n c -> f (c n)")
        audio_length, _ = feature_array.shape
        selected_feature = []
        selected_idx = []
        aud_vid_ratio = int(self.aud_feature_fps / self.vid_fps)
        assert aud_vid_ratio == 2, "only implemented for aud_vid_ratio == 2"
        center_idx = vid_idx * aud_vid_ratio

        if is_first_token:
            left_idx = center_idx - (4 - 1 + self.audio_feat_window_size) * aud_vid_ratio
            right_idx = center_idx + (1 + self.audio_feat_window_size) * aud_vid_ratio
        else:
            left_idx = center_idx - self.audio_feat_window_size * aud_vid_ratio
            right_idx = center_idx + (4 + self.audio_feat_window_size) * aud_vid_ratio

        for idx in range(left_idx, right_idx):
            idx = min(audio_length - 1, max(0, idx))
            selected_feature.append(feature_array[idx])
            selected_idx.append(idx)

        selected_feature = torch.stack(selected_feature)
        return selected_feature, selected_idx

    def crop_overlap_audio_window(self, audio_feat, start_index):
        # output: token_frames, audio_window, c
        selected_feature_list = []
        for i in range(self.token_frames):
            if i == 0:
                vid_center_idx = start_index
                selected_feature, _ = self.get_sliced_feature(
                    feature_array=audio_feat,
                    vid_idx=vid_center_idx,
                    is_first_token=True,
                )
            else:
                vid_center_idx = start_index + (i - 1) * 4 + 1
                selected_feature, _ = self.get_sliced_feature(
                    feature_array=audio_feat,
                    vid_idx=vid_center_idx,
                    is_first_token=False,
                )
            selected_feature_list.append(selected_feature)
        return torch.stack(selected_feature_list)


class WhisperProcessor(AudioBaseProcessor):
    def __init__(
        self,
        model_path=DEFAULT_WHISPER_PATH,
        device="cpu",
        num_frames=77,
        audio_feat_window_size=0,
        vid_fps=25,
        aud_feature_fps=50,
        embedding_num_layers=1,
        sample_rate=16000,
    ):
        super().__init__(
            num_frames=num_frames,
            audio_feat_window_size=audio_feat_window_size,
            vid_fps=vid_fps,
            aud_feature_fps=aud_feature_fps,
            embedding_num_layers=embedding_num_layers,
            sample_rate=sample_rate,
        )
        self.model = load_model(model_path, device)
        self.model_type = model_path.split("/")[-1].split(".")[0]

    @torch.no_grad()
    def _audio2feat(self, audio_path: str):
        # output: f, l, c
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result["segments"]:
            encoder_embeddings = emb["encoder_embeddings"]
            encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb["start"])
            end_idx = int(emb["end"])
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = torch.from_numpy(np.concatenate(embed_list, axis=0))
        concatenated_array = concatenated_array.cpu().detach()
        concatenated_array = concatenated_array.to(dtype=torch.float32)
        return concatenated_array

    def audio2feat(self, audio_path, use_cache=False):
        # cached feature: f, l, c
        if not use_cache:
            audio_feat = self._audio2feat(audio_path)
            audio_feat = resample_audio_feature(audio_feat, self.aud_feature_fps, 50)
            return audio_feat

        wav2vec_path = get_audio_feature_cache_path(audio_path, self.model_type)
        if os.path.isfile(wav2vec_path):
            try:
                audio_feat = np.load(wav2vec_path).astype(np.float32)
                audio_feat = torch.from_numpy(audio_feat)
            except Exception as e:
                print(f"{type(e).__name__} - {e} - {wav2vec_path}")
                if os.path.exists(wav2vec_path):
                    os.remove(wav2vec_path)
                audio_feat = self._audio2feat(audio_path)
                np.save(wav2vec_path, audio_feat.cpu().numpy())
        else:
            print(f"Caching to {wav2vec_path}")
            audio_feat = self._audio2feat(audio_path)
            np.save(wav2vec_path, audio_feat.cpu().numpy())
        audio_feat = resample_audio_feature(audio_feat, self.aud_feature_fps, 50)
        return audio_feat


class Wav2VecProcessor(AudioBaseProcessor):
    def __init__(
        self,
        model_path=DEFAULT_WAV2VEC_PATH,
        device="cpu",
        num_frames=77,
        audio_feat_window_size=0,
        vid_fps=25,
        aud_feature_fps=50,
        embedding_num_layers=1,
        sample_rate=16000,
    ):
        super().__init__(
            num_frames=num_frames,
            audio_feat_window_size=audio_feat_window_size,
            vid_fps=vid_fps,
            aud_feature_fps=aud_feature_fps,
            embedding_num_layers=embedding_num_layers,
            sample_rate=sample_rate,
        )

        try:
            self.model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True).to(device=device)
        except Exception as e:
            print(f"{type(e).__name__} - {e} - {model_path}")
            print(f"Failed to load wav2vec model from {model_path}. Model will be downloaded from huggingface.")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device=device)

        self.model.feature_extractor._freeze_parameters()
        self.model_type = model_path.split("/")[-1].split(".")[0]
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)

    @torch.no_grad()
    def _audio2feat(self, audio_path: str):
        # output: f, l, c
        if isinstance(audio_path, np.ndarray):
            audio = audio_path
        else:
            audio = load_audio(audio_path)

        audio_feature = np.squeeze(self.wav2vec_feature_extractor(audio, sampling_rate=self.sample_rate).input_values)
        audio_feature = torch.from_numpy(audio_feature).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            embeddings = self.model(audio_feature, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"
        embeddings = embeddings.hidden_states
        embeddings = torch.stack(embeddings, dim=1).squeeze(0)
        embeddings = rearrange(embeddings, "l f c -> f l c")
        embeddings = embeddings.cpu().detach()
        embeddings = embeddings.to(dtype=torch.float32)
        return embeddings

    def audio2feat(self, audio_path, use_cache=False):
        # cached feature: f, l, c
        if not use_cache:
            audio_feat = self._audio2feat(audio_path)
            audio_feat = resample_audio_feature(audio_feat, self.aud_feature_fps, 50)
            return audio_feat

        wav2vec_path = get_audio_feature_cache_path(audio_path, self.model_type)
        if os.path.isfile(wav2vec_path):
            try:
                audio_feat = np.load(wav2vec_path).astype(np.float32)
                audio_feat = torch.from_numpy(audio_feat)
            except Exception as e:
                print(f"{type(e).__name__} - {e} - {wav2vec_path}")
                if os.path.exists(wav2vec_path):
                    os.remove(wav2vec_path)
                audio_feat = self._audio2feat(audio_path)
                np.save(wav2vec_path, audio_feat.cpu().numpy())
        else:
            print(f"Caching to {wav2vec_path}")
            audio_feat = self._audio2feat(audio_path)
            np.save(wav2vec_path, audio_feat.cpu().numpy())
        audio_feat = resample_audio_feature(audio_feat, self.aud_feature_fps, 50)
        return audio_feat
