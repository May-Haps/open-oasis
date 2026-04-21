from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import av
import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_model
from torch import autocast
from torchvision.transforms.functional import resize
from tqdm import tqdm

from dit.utils import ACTION_KEYS, one_hot_actions
from dit.vae import VAE_models

TARGET_SIZE = [360, 640]
LATENT_SCALE = 0.07843137255
ACTION_DIM = 25
CAMERA_CLIP_DEGREES = 20.0


RAW_MINERL_ACTION_KEYS = {
    "forward",
    "left",
    "back",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
}


def load_actions_tensor(actions_path: Path) -> torch.Tensor:
    path_str = str(actions_path)
    if path_str.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(actions_path))
    elif path_str.endswith(".one_hot_actions.pt"):
        actions = torch.load(actions_path, weights_only=True)
    elif path_str.endswith(".npz"):
        actions = load_minerl_npz_actions(actions_path)
    else:
        raise ValueError(f"Unsupported actions file: {actions_path}")

    actions = actions.float()
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shaped [T, {ACTION_DIM}], got {tuple(actions.shape)}")
    return actions


def load_minerl_npz_actions(actions_path: Path) -> torch.Tensor:
    payload = np.load(actions_path, allow_pickle=True)
    action_length = None

    for key in payload.files:
        if key.startswith("action$"):
            action_length = len(payload[key])
            break

    if action_length is None:
        raise ValueError(f"No action arrays found in {actions_path}")

    actions = torch.zeros(action_length, ACTION_DIM, dtype=torch.float32)

    for idx, action_key in enumerate(ACTION_KEYS):
        if action_key == "cameraX":
            camera = np.asarray(payload["action$camera"], dtype=np.float32)
            actions[:, idx] = torch.from_numpy(
                np.clip(camera[:, 0], -CAMERA_CLIP_DEGREES, CAMERA_CLIP_DEGREES) / CAMERA_CLIP_DEGREES
            )
        elif action_key == "cameraY":
            camera = np.asarray(payload["action$camera"], dtype=np.float32)
            actions[:, idx] = torch.from_numpy(
                np.clip(camera[:, 1], -CAMERA_CLIP_DEGREES, CAMERA_CLIP_DEGREES) / CAMERA_CLIP_DEGREES
            )
        elif action_key in RAW_MINERL_ACTION_KEYS:
            values = np.asarray(payload[f"action${action_key}"], dtype=np.float32)
            actions[:, idx] = torch.from_numpy(np.clip(values, 0.0, 1.0))

    return actions


def _encode_buffer(buffer, vae, device):
    frames = torch.stack(buffer, dim=0)
    frames = rearrange(frames, "t h w c -> t c h w").float() / 255.0
    frames = resize(frames, TARGET_SIZE, antialias=True).to(device)

    latent_h = TARGET_SIZE[0] // vae.patch_size
    latent_w = TARGET_SIZE[1] // vae.patch_size

    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        latents = vae.encode(frames * 2 - 1).mean * LATENT_SCALE

    latents = rearrange(latents, "b (h w) c -> b c h w", h=latent_h, w=latent_w)
    return latents.cpu()


def encode_video_streaming(video_path, vae, device, batch_size):
    all_latents = []
    buffer = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            buffer.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if len(buffer) >= batch_size:
                all_latents.append(_encode_buffer(buffer, vae, device))
                buffer = []
        if buffer:
            all_latents.append(_encode_buffer(buffer, vae, device))
    return torch.cat(all_latents, dim=0)


def preprocess_episode(
    video_path: Path,
    actions_path: Path,
    out_dir: Path,
    vae: torch.nn.Module,
    device: str,
    batch_size: int,
) -> dict:
    if video_path.stem == "recording" and actions_path.name == "rendered.npz":
        episode_id = video_path.parent.name
    else:
        episode_id = video_path.stem
    episode_out = out_dir / episode_id
    episode_out.mkdir(parents=True, exist_ok=True)
    
    actions = load_actions_tensor(actions_path)

    latents = encode_video_streaming(
        video_path,
        vae,
        device=device,
        batch_size=batch_size
    ).half()

    num_frames = latents.shape[0]

    # Raw MineRL trajectories store extra video frames; keep the first next-state-aligned segment.
    if str(actions_path).endswith(".npz") and num_frames >= len(actions) + 1:
        latents = latents[: len(actions) + 1]
        num_frames = latents.shape[0]

    # Store transition actions so each clip can prepend its own zero-action prompt frame.
    if len(actions) == num_frames:
        actions = actions[:-1]

    if len(actions) != num_frames - 1:
        raise ValueError(
            f"Alignment mismatch for {episode_id}: "
            f"{num_frames} frames vs {len(actions)} actions; expected actions = frames - 1"
        )

    latents_path = episode_out / "latents.pt"
    actions_out_path = episode_out / "actions.one_hot.pt"
    meta_path = episode_out / "meta.json"

    torch.save(latents, latents_path)
    torch.save(actions, actions_out_path)

    meta = {
        "episode_id": episode_id,
        "video_path": str(video_path),
        "actions_path": str(actions_path),
        "latents_path": str(latents_path),
        "processed_actions_path": str(actions_out_path),
        "frame_count": int(num_frames),
        "action_count": int(len(actions)),
        "latent_shape": list(latents.shape),
        "action_shape": list(actions.shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    import os
    pairs = []
    mp4_paths = []
    for root, _, files in os.walk(input_dir, followlinks=True):
        for name in files:
            if name.endswith(".mp4"):
                mp4_paths.append(Path(root) / name)
    for video_path in sorted(mp4_paths):
        stem = video_path.with_suffix("")
        preferred = [
            Path(str(stem) + ".one_hot_actions.pt"),
            Path(str(stem) + ".actions.pt"),
            video_path.parent / "rendered.npz",
        ]
        match = next((candidate for candidate in preferred if candidate.exists()), None)
        if match is not None:
            pairs.append((video_path, match))
    return pairs


def load_vae(vae_ckpt: str, device: str) -> torch.nn.Module:
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    if vae_ckpt.endswith(".safetensors"):
        load_model(vae, vae_ckpt)
    else:
        state = torch.load(vae_ckpt, weights_only=True)
        vae.load_state_dict(state)
    return vae.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(args.vae_ckpt, args.device)
    pairs = find_pairs(input_dir)
    if not pairs:
        raise ValueError(f"No video/action pairs found under {input_dir}")

    manifest = []
    for video_path, actions_path in tqdm(pairs, desc="Episodes"):
        manifest.append(
            preprocess_episode(
                video_path=video_path,
                actions_path=actions_path,
                out_dir=output_dir,
                vae=vae,
                device=args.device,
                batch_size=args.batch_size,
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
