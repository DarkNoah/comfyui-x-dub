import argparse
import os
from dataclasses import dataclass

import torch
from PIL import Image

from diffsynth.pipelines.lip_sync import LipSyncPipeline, ModelConfig
from lip_sync_preprocess import preprocess_video_with_dwpose
from utils import (
    blend_crop_video_with_ref,
    color_correction,
    concat_pil_videos_horizontally,
    get_total_length,
    make_pingpong_indices,
    paste_video_back,
    save_video_with_audio,
)

CLIP_NUM_FRAMES = 77
MOTION_NUM_FRAMES = 5

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_DIT_PATH = os.path.join(CHECKPOINTS_DIR, "X-Dub_model.safetensors")
DEFAULT_TEXT_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "models_t5_umt5-xxl-enc-bf16.safetensors")
DEFAULT_VAE_PATH = os.path.join(CHECKPOINTS_DIR, "Wan2.2_VAE.safetensors")
DEFAULT_TOKENIZER_PATH = os.path.join(CHECKPOINTS_DIR, "umt5-xxl")
DEFAULT_WHISPER_PATH = os.path.join(CHECKPOINTS_DIR, "whisper", "large-v2.pt")
DEFAULT_WAV2VEC_PATH = os.path.join(CHECKPOINTS_DIR, "wav2vec2-base-960h")

vram_config = {
    # "offload_dtype": "disk",
    # "offload_device": "disk",
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

@dataclass
class PreprocessedLipSyncSample:
    raw_video: list[Image.Image]
    ref_video: list[Image.Image]
    bboxes: list[list[int]]
    audio_path: str


def smooth_transition_latent(previous_segment, current_segment):
    boundary_latent = 0.5 * (previous_segment[:, :, -1:, :, :] + current_segment[:, :, 1:2, :, :])
    previous_segment = previous_segment.clone()
    previous_segment[:, :, -1:, :, :] = boundary_latent
    return previous_segment, current_segment[:, :, 2:, :, :]


def preprocess_inputs(video_path: str, audio_path: str, args) -> PreprocessedLipSyncSample:
    sample_name = args.sample_name or os.path.splitext(os.path.basename(video_path))[0]
    raw_video, ref_video, bboxes, case_flag = preprocess_video_with_dwpose(
        video_path,
        output_dir=args.output_dir,
        sample_name=sample_name,
    )
    print(
        f"[Preprocess {sample_name}] "
        f"num_raw_frames={len(raw_video)}, num_ref_frames={len(ref_video)}, case_flag={case_flag}"
    )
    print(f"[Preprocess {sample_name}] first_bbox={bboxes[0]}")
    return PreprocessedLipSyncSample(
        raw_video=raw_video,
        ref_video=ref_video,
        bboxes=bboxes,
        audio_path=audio_path,
    )


def validate_preprocessed_sample(sample: PreprocessedLipSyncSample):
    assert os.path.exists(sample.audio_path), f"audio path not found: {sample.audio_path}"
    assert len(sample.ref_video) > 0, "ref_video cannot be empty."
    assert len(sample.raw_video) == len(sample.ref_video), "raw_video and ref_video must have the same length."
    assert len(sample.bboxes) == len(sample.ref_video), "bboxes and ref_video must have the same length."
    for bbox in sample.bboxes:
        assert len(bbox) == 4, f"bbox must have 4 ints, got: {bbox}"
    return sample


def infer_one_sample(pipe, sample: PreprocessedLipSyncSample, sample_name: str, args):
    sample = validate_preprocessed_sample(sample)
    raw_video = sample.raw_video
    ref_video = sample.ref_video
    bboxes = sample.bboxes
    audio_path = sample.audio_path

    print(f"[Sample {sample_name}] video_path={args.video_path}")
    print(f"[Sample {sample_name}] audio_path={audio_path}")
    print(f"[Sample {sample_name}] num_ref_frames={len(ref_video)}")

    total_length = get_total_length(audio_path, clip_num_frames=CLIP_NUM_FRAMES, motion_num_frames=MOTION_NUM_FRAMES)
    num_clips = 1 + max(0, total_length - CLIP_NUM_FRAMES) // (CLIP_NUM_FRAMES - MOTION_NUM_FRAMES)
    pingpong_indices = make_pingpong_indices(len(ref_video), total_length)
    ref_video = [ref_video[index] for index in pingpong_indices]
    raw_video = [raw_video[index] for index in pingpong_indices]
    bboxes = [bboxes[index] for index in pingpong_indices]

    print(f"[Sample {sample_name}] total_length={total_length}, num_clips={num_clips}")

    motion_video = None
    whisper_feat = None
    wav2vec_feat = None
    latents_segments = []
    start_idx = 0

    for clip_idx in range(num_clips):
        print(f"[Sample {sample_name}] [{clip_idx + 1}/{num_clips}] start_idx={start_idx}")
        ref_video_clip = ref_video[start_idx: start_idx + CLIP_NUM_FRAMES]
        output_video, outputs = pipe(
            ref_video=ref_video_clip,
            start_idx=start_idx,
            audio_npy_path=None,
            audio_wav_path=audio_path,
            whisper_feat=whisper_feat,
            wav2vec_feat=wav2vec_feat,
            prompt="",
            motion_video=motion_video,
            height=512,
            width=512,
            num_frames=CLIP_NUM_FRAMES,
            motion_latents_num_frames=2,
            ref_cfg_scale=args.ref_cfg_scale,
            audio_cfg_scale=args.audio_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            use_dynamic_cfg=True,
            replace_border_latents=True,
            replace_border_latents_width=1,
        )

        if whisper_feat is None:
            whisper_feat = outputs.get("whisper_feat", None)
        if wav2vec_feat is None:
            wav2vec_feat = outputs.get("wav2vec_feat", None)

        output_video = color_correction(output_video, ref_video_clip)
        motion_video = output_video[-MOTION_NUM_FRAMES:]
        output_latents = outputs["latents"]

        if clip_idx == 0:
            latents_segments.append(output_latents)
        else:
            latents_segments[-1], output_latents_to_append = smooth_transition_latent(latents_segments[-1], output_latents)
            latents_segments.append(output_latents_to_append)

        start_idx += CLIP_NUM_FRAMES - MOTION_NUM_FRAMES
        

    output_latents = torch.cat(latents_segments, dim=2)
    pipe.load_models_to_device(["vae"])
    final_video = pipe.vae.decode(output_latents, device=pipe.device, tiled=False, tile_size=(32, 32), tile_stride=(16, 16))
    final_output_video = pipe.vae_output_to_video(final_video)[:total_length]
    final_output_video = color_correction(final_output_video, ref_video[:total_length])

    blended_crop_video = blend_crop_video_with_ref(
        final_output_video,
        ref_video[:total_length],
        replace_border_latents_width=1,
    )
    final_pasted_video = paste_video_back(blended_crop_video, raw_video, bboxes)

    crop_compare_video = concat_pil_videos_horizontally(ref_video[:total_length], final_output_video)
    save_video_with_audio(crop_compare_video, audio_path, f"{sample_name}_crop_compare", args.output_dir)

    paste_compare_video = concat_pil_videos_horizontally(raw_video[:total_length], final_pasted_video)
    save_video_with_audio(paste_compare_video, audio_path, f"{sample_name}_paste_compare", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--ref_cfg_scale", type=float, default=2.5)
    parser.add_argument("--audio_cfg_scale", type=float, default=10.0)
    parser.add_argument("--audio_feat_window_size", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_DIT_PATH)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_name = args.sample_name or os.path.splitext(os.path.basename(args.video_path))[0]
    sample = preprocess_inputs(args.video_path, args.audio_path, args)

    pipe = LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.ckpt_path, **vram_config),
            ModelConfig(path=DEFAULT_TEXT_ENCODER_PATH, **vram_config),
            ModelConfig(path=DEFAULT_VAE_PATH, **vram_config),
        ],
        tokenizer_config=ModelConfig(path=DEFAULT_TOKENIZER_PATH),
        args=args,
        whisper_ckpt_path=DEFAULT_WHISPER_PATH,
        wav2vec_ckpt_path=DEFAULT_WAV2VEC_PATH,
    )
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    infer_one_sample(pipe, sample, sample_name, args)

    
if __name__ == "__main__":
    main()
