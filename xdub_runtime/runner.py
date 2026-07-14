import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PLUGIN_DIR / "xdub_runtime" / "source"
sys.path.insert(0, str(SOURCE_DIR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--dit-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--ref-cfg-scale", type=float, default=2.5)
    parser.add_argument("--audio-cfg-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_source_paths(models):
    import lip_sync_preprocess

    lip_sync_preprocess.DET_ONNX_PATH = str(models["dwpose_detector"])
    lip_sync_preprocess.POSE_ONNX_PATH = str(models["dwpose_pose"])


def log(message):
    print(f"[X-Dub {time.strftime('%H:%M:%S')}] {message}", flush=True)


def console_progress(iterable):
    total = len(iterable)
    for index, item in enumerate(iterable, start=1):
        log(f"Denoising step {index}/{total}")
        yield item


def main():
    args = parse_args()
    log("Resolving runtime model paths...")
    from model_paths import resolve_models
    models = resolve_models()
    configure_source_paths(models)

    import torch
    from diffsynth.pipelines.lip_sync import LipSyncPipeline, ModelConfig
    import infer_lip_sync_pipeline as original
    from utils import (
        blend_crop_video_with_ref,
        color_correction,
        get_total_length,
        make_pingpong_indices,
        paste_video_back,
        save_video_with_audio,
    )

    run_dir = Path(args.output_path).parent / f".{Path(args.output_path).stem}_runtime"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_args = SimpleNamespace(
        video_path=args.video_path,
        audio_path=args.audio_path,
        sample_name="result",
        output_dir=str(run_dir),
        ref_cfg_scale=args.ref_cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        audio_feat_window_size=0,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        ckpt_path=args.dit_path,
    )

    log("Preprocessing video, audio, face detection, and pose...")
    sample = original.preprocess_inputs(args.video_path, args.audio_path, source_args)
    sample = original.validate_preprocessed_sample(sample)
    total_length = get_total_length(args.audio_path, original.CLIP_NUM_FRAMES, original.MOTION_NUM_FRAMES)
    log(f"Preprocessing complete: {total_length} output frames at 25 FPS.")
    indices = make_pingpong_indices(len(sample.ref_video), total_length)
    ref_video = [sample.ref_video[i] for i in indices]
    raw_video = [sample.raw_video[i] for i in indices]
    bboxes = [sample.bboxes[i] for i in indices]

    log("Loading X-Dub, text encoder, VAE, and audio encoders...")
    pipe = LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.dit_path, **original.vram_config),
            ModelConfig(path=str(models["text_encoder"]), **original.vram_config),
            ModelConfig(path=args.vae_path, **original.vram_config),
        ],
        tokenizer_config=ModelConfig(path=str(models["tokenizer"])),
        args=source_args,
        whisper_ckpt_path=str(models["whisper"]),
        wav2vec_ckpt_path=str(models["wav2vec"]),
    )
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    log("Models loaded.")

    num_clips = 1 + max(0, total_length - original.CLIP_NUM_FRAMES) // (original.CLIP_NUM_FRAMES - original.MOTION_NUM_FRAMES)
    motion_video = whisper_feat = wav2vec_feat = None
    segments = []
    start_idx = 0
    log(f"Starting diffusion: {num_clips} clip(s), {args.num_inference_steps} step(s) each.")
    for clip_idx in range(num_clips):
        clip_started = time.monotonic()
        clip = ref_video[start_idx:start_idx + original.CLIP_NUM_FRAMES]
        log(
            f"Clip {clip_idx + 1}/{num_clips}: frames {start_idx + 1}-"
            f"{min(start_idx + original.CLIP_NUM_FRAMES, total_length)}..."
        )
        output_video, outputs = pipe(
            ref_video=clip,
            start_idx=start_idx,
            audio_npy_path=None,
            audio_wav_path=args.audio_path,
            whisper_feat=whisper_feat,
            wav2vec_feat=wav2vec_feat,
            prompt="",
            motion_video=motion_video,
            height=512,
            width=512,
            num_frames=original.CLIP_NUM_FRAMES,
            motion_latents_num_frames=2,
            ref_cfg_scale=args.ref_cfg_scale,
            audio_cfg_scale=args.audio_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            use_dynamic_cfg=True,
            replace_border_latents=True,
            replace_border_latents_width=1,
            progress_bar_cmd=console_progress,
        )
        whisper_feat = outputs.get("whisper_feat") if whisper_feat is None else whisper_feat
        wav2vec_feat = outputs.get("wav2vec_feat") if wav2vec_feat is None else wav2vec_feat
        output_video = color_correction(output_video, clip)
        motion_video = output_video[-original.MOTION_NUM_FRAMES:]
        latents = outputs["latents"]
        if segments:
            segments[-1], latents = original.smooth_transition_latent(segments[-1], latents)
        segments.append(latents)
        log(f"Clip {clip_idx + 1}/{num_clips} complete in {time.monotonic() - clip_started:.1f}s.")
        start_idx += original.CLIP_NUM_FRAMES - original.MOTION_NUM_FRAMES

    log("Diffusion complete. Decoding VAE latents...")
    output_latents = torch.cat(segments, dim=2)
    pipe.load_models_to_device(["vae"])
    decoded = pipe.vae.decode(output_latents, device=pipe.device, tiled=False, tile_size=(32, 32), tile_stride=(16, 16))
    generated = pipe.vae_output_to_video(decoded)[:total_length]
    log("VAE decode complete. Applying color correction and compositing...")
    generated = color_correction(generated, ref_video[:total_length])
    generated = blend_crop_video_with_ref(generated, ref_video[:total_length], replace_border_latents_width=1)
    final_frames = paste_video_back(generated, raw_video, bboxes)
    log("Compositing complete. Encoding final video with audio...")
    save_video_with_audio(final_frames, args.audio_path, "result", str(run_dir))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir / "result.mp4"), output_path)
    shutil.rmtree(run_dir, ignore_errors=True)
    log(f"Finished: {output_path}")


if __name__ == "__main__":
    main()
