from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from data.dataset_coinrun_streaming import CoinRunStreamingDataset
from model.dit import CoinRunWorldModel, CoinRunWorldModelSmall
from model.utils import sigmoid_beta_schedule
from training.noise_scheduler import NoiseScheduler


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def try_build_pyiqa_metric(name: str, device: str):
    try:
        import pyiqa  # type: ignore

        return pyiqa.create_metric(name, device=device)
    except Exception as exc:
        print(f"{name} unavailable: {exc}")
        return None


class ClipEmbedder:
    def __init__(self, device: str) -> None:
        try:
            from transformers import AutoProcessor, CLIPModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for CLIP-based temporal consistency."
            ) from exc

        self.device = device
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    @torch.no_grad()
    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        images = [frame.permute(1, 2, 0).cpu().numpy() for frame in frames]
        proc = self.processor(images=images, return_tensors="pt")
        proc = {key: value.to(self.device) for key, value in proc.items()}
        with autocast_context(self.device):
            feats = self.model.get_image_features(**proc)
        if hasattr(feats, "image_embeds"):
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif isinstance(feats, (tuple, list)):
            feats = feats[0]
        return F.normalize(feats.float(), dim=-1)


def infer_coinrun_model(state_dict: dict[str, torch.Tensor], ckpt_config: dict) -> nn.Module:
    hidden_size = int(state_dict["x_embedder.proj.weight"].shape[0])
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    depth = len(block_ids)
    action_cond_mode = ckpt_config.get("action_cond_mode", "linear")

    if hidden_size == 512 and depth == 6:
        return CoinRunWorldModelSmall(
            external_cond_mode=action_cond_mode,
        )
    if hidden_size == 640 and depth == 8:
        return CoinRunWorldModel(
            external_cond_mode=action_cond_mode,
        )

    raise ValueError(
        "Unsupported CoinRun checkpoint architecture. "
        f"Observed hidden_size={hidden_size}, depth={depth}."
    )


def load_checkpoint_model(ckpt_path: str, device: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint must be a training checkpoint with a top-level 'model' key.")

    ckpt_config = ckpt.get("config", {})
    state_dict = ckpt["model"]
    model = infer_coinrun_model(state_dict, ckpt_config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, ckpt_config


def build_dataloader(
    data_dir: str,
    clip_len: int,
    stride: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CoinRunStreamingDataset(data_dir, clip_len=clip_len, stride=stride, seed=0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


@torch.no_grad()
def generate_rollout(
    model: nn.Module,
    prompt_frames: torch.Tensor,
    actions: torch.Tensor,
    noise_scheduler_alphas: torch.Tensor,
    device: str,
    ddim_steps: int = 10,
    total_frames: int = 16,
    n_prompt: int = 1,
    noise_abs_max: float = 20.0,
    stabilization_level: int = 15,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    x = prompt_frames.to(device) * 2 - 1
    actions = actions.to(device)
    batch_size = x.shape[0]
    noise_range = torch.linspace(-1, noise_scheduler_alphas.shape[0] - 1, ddim_steps + 1, device=device)

    for frame_idx in range(n_prompt, total_frames):
        chunk = torch.randn((batch_size, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, frame_idx + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((batch_size, frame_idx), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((batch_size, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((batch_size, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)[:, start_frame:]
            t_next = torch.cat([t_ctx, t_next], dim=1)[:, start_frame:]

            x_curr = x[:, start_frame:]
            with autocast_context(device):
                v = model(x_curr, t, actions[:, start_frame : frame_idx + 1])

            ac = noise_scheduler_alphas[t]
            x_start = ac.sqrt() * x_curr - (1 - ac).sqrt() * v
            x_noise = ((1 / ac).sqrt() * x_curr - x_start) / (1 / ac - 1).sqrt()

            an = noise_scheduler_alphas[t_next]
            an[:, :-1] = 1.0
            if noise_idx == 1:
                an[:, -1:] = 1.0
            x[:, -1:] = (an.sqrt() * x_start + x_noise * (1 - an).sqrt())[:, -1:]

    out = (x.clamp(-1, 1) + 1) / 2
    out = rearrange(out, "b t c h w -> b t h w c")
    out = (out * 255).byte().cpu()

    if was_training:
        model.train()
    return out


def save_rollout_video(path: Path, frames: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), frames, fps=fps)


@torch.no_grad()
def evaluate_noise_loss_and_recon(
    model: nn.Module,
    loader: DataLoader,
    noise_scheduler: NoiseScheduler,
    device: str,
    max_noise_level: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    gt_frames: list[np.ndarray] = []
    pred_frames: list[np.ndarray] = []

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "t -> t 1 1 1")

    for batch_idx, batch in enumerate(tqdm(loader, desc="Val batches", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x0 = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        batch_size, timesteps = x0.shape[:2]

        t = torch.randint(0, noise_scheduler.timesteps, (batch_size, timesteps), device=device)
        noise = torch.randn_like(x0)
        x0_scaled = x0 * 2 - 1
        xt, v_target = noise_scheduler.noised_sample_and_velocity_target(x0_scaled, t, noise)

        with autocast_context(device):
            v_pred = model(xt, t, actions)

        loss = nn.functional.mse_loss(v_pred.float(), v_target.float())
        loss_total += float(loss.item()) * batch_size
        sample_count += batch_size

        ac = alphas_cumprod[t]
        x0_pred = ac.sqrt() * xt - (1 - ac).sqrt() * v_pred.float()
        x0_pred = ((x0_pred.clamp(-1, 1) + 1) / 2 * 255).byte()
        x0_pred = x0_pred.permute(0, 1, 3, 4, 2).reshape(-1, 64, 64, 3).cpu().numpy()
        x0_gt = (x0 * 255).byte().permute(0, 1, 3, 4, 2).reshape(-1, 64, 64, 3).cpu().numpy()
        gt_frames.append(x0_gt.astype(np.float32))
        pred_frames.append(x0_pred.astype(np.float32))

    if sample_count == 0:
        raise ValueError("No validation batches were evaluated.")

    gt_all = np.concatenate(gt_frames, axis=0)
    pred_all = np.concatenate(pred_frames, axis=0)
    psnr = peak_signal_noise_ratio(gt_all, pred_all, data_range=255)
    ssim = float(
        np.mean(
            [
                structural_similarity(gt_all[i], pred_all[i], data_range=255, channel_axis=-1)
                for i in range(len(gt_all))
            ]
        )
    )

    return {
        "noise_loss": loss_total / sample_count,
        "psnr": float(psnr),
        "ssim": ssim,
        "samples": float(sample_count),
    }


@torch.no_grad()
def evaluate_rollouts(
    model: nn.Module,
    dataset: CoinRunStreamingDataset,
    device: str,
    max_noise_level: int,
    ddim_steps: int,
    n_prompt_frames: int,
    rollout_frames: int,
    num_samples: int,
    save_dir: Path | None,
) -> dict[str, float | None]:
    if num_samples <= 0:
        return {
            "rollout_psnr": None,
            "rollout_ssim": None,
            "musiq": None,
            "laion_aes": None,
            "temporal_consistency": None,
        }

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas_cumprod = rearrange(torch.cumprod(1.0 - betas, dim=0), "t -> t 1 1 1")

    psnrs: list[float] = []
    ssims: list[float] = []
    musiqs: list[float] = []
    laion_scores: list[float] = []
    temporal_scores: list[float] = []

    musiq_metric = try_build_pyiqa_metric("musiq", device)
    laion_metric = try_build_pyiqa_metric("laion_aes", device)
    clip_embedder = None
    try:
        clip_embedder = ClipEmbedder(device)
    except Exception as exc:
        print(f"temporal consistency unavailable: {exc}")

    for sample_idx, sample in enumerate(tqdm(dataset, total=num_samples, desc="Rollouts", leave=False)):
        if sample_idx >= num_samples:
            break

        prompt_frames = sample["frames"][:n_prompt_frames].unsqueeze(0)
        gt_frames = sample["frames"][:rollout_frames].unsqueeze(0)
        actions = sample["actions"][:rollout_frames].unsqueeze(0)
        generated = generate_rollout(
            model=model,
            prompt_frames=prompt_frames,
            actions=actions,
            noise_scheduler_alphas=alphas_cumprod,
            device=device,
            ddim_steps=ddim_steps,
            total_frames=rollout_frames,
            n_prompt=n_prompt_frames,
        )
        gt_uint8 = (rearrange(gt_frames, "b t c h w -> b t h w c") * 255).byte().cpu()

        pred_future = generated[:, n_prompt_frames:].numpy().astype(np.float32)
        gt_future = gt_uint8[:, n_prompt_frames:].numpy().astype(np.float32)
        pred_future_tensor = rearrange(
            generated[:, n_prompt_frames:].float().div(255.0),
            "b t h w c -> (b t) c h w",
        )

        psnrs.append(peak_signal_noise_ratio(gt_future, pred_future, data_range=255))
        ssims.append(
            float(
                np.mean(
                    [
                        structural_similarity(
                            gt_future[0, frame_idx],
                            pred_future[0, frame_idx],
                            data_range=255,
                            channel_axis=-1,
                        )
                        for frame_idx in range(pred_future.shape[1])
                    ]
                )
            )
        )

        if musiq_metric is not None:
            musiqs.append(float(musiq_metric(pred_future_tensor.to(device)).mean().item()))

        if laion_metric is not None:
            laion_scores.append(float(laion_metric(pred_future_tensor.to(device)).mean().item()))

        if clip_embedder is not None and pred_future_tensor.shape[0] > 1:
            emb = clip_embedder.embed(pred_future_tensor)
            temporal_scores.append(float(F.cosine_similarity(emb[1:], emb[:-1], dim=-1).mean().item()))

        if save_dir is not None:
            save_rollout_video(save_dir / f"rollout_{sample_idx:03d}.mp4", generated[0], fps=15)

    if not psnrs:
        return {
            "rollout_psnr": None,
            "rollout_ssim": None,
            "musiq": None,
            "laion_aes": None,
            "temporal_consistency": None,
        }

    return {
        "rollout_psnr": float(np.mean(psnrs)),
        "rollout_ssim": float(np.mean(ssims)),
        "musiq": float(np.mean(musiqs)) if musiqs else None,
        "laion_aes": float(np.mean(laion_scores)) if laion_scores else None,
        "temporal_consistency": float(np.mean(temporal_scores)) if temporal_scores else None,
    }


def main(args: argparse.Namespace) -> None:
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Validation directory not found: {args.data_dir}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to run on CPU.")

    model, ckpt_config = load_checkpoint_model(args.ckpt, args.device)
    noise_scheduler = NoiseScheduler(args.max_noise_level, args.device)
    loader = build_dataloader(
        data_dir=args.data_dir,
        clip_len=args.clip_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    metrics = evaluate_noise_loss_and_recon(
        model=model,
        loader=loader,
        noise_scheduler=noise_scheduler,
        device=args.device,
        max_noise_level=args.max_noise_level,
        max_batches=args.max_batches,
    )

    rollout_dir = Path(args.save_dir) / "rollouts" if args.save_dir else None
    rollout_metrics = evaluate_rollouts(
        model=model,
        dataset=CoinRunStreamingDataset(args.data_dir, clip_len=args.clip_len, stride=args.stride, seed=0),
        device=args.device,
        max_noise_level=args.max_noise_level,
        ddim_steps=args.ddim_steps,
        n_prompt_frames=args.n_prompt_frames,
        rollout_frames=args.rollout_frames,
        num_samples=args.num_rollout_samples,
        save_dir=rollout_dir,
    )
    metrics.update(rollout_metrics)
    metrics["checkpoint"] = args.ckpt
    metrics["data_dir"] = args.data_dir
    metrics["action_cond_mode"] = ckpt_config.get("action_cond_mode", "linear")

    print(json.dumps(metrics, indent=2))

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {save_path / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a CoinRun world-model checkpoint on the validation split.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a CoinRun training checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data/coinrun/val", help="CoinRun validation shard directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-noise-level", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit validation batches for a quicker eval.")
    parser.add_argument("--num-rollout-samples", type=int, default=4)
    parser.add_argument("--n-prompt-frames", type=int, default=1)
    parser.add_argument("--rollout-frames", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=None)
    main(parser.parse_args())
