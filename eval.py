from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from torchvision.io import write_video
from torchvision.transforms.functional import resize
from tqdm import tqdm

from dataset import ProcessedGameDataset
from dit import DiT, DiT_models
from utils import sigmoid_beta_schedule
from vae import VAE_models

VAE_TARGET_SIZE = (360, 640)
SCALING_FACTOR = 0.07843137255


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class NoiseScheduler:
    def __init__(self, timesteps: int, device: str) -> None:
        self.timesteps = timesteps
        beta_ts = sigmoid_beta_schedule(timesteps).to(device=device, dtype=torch.float32)
        alpha_ts = 1.0 - beta_ts
        alpha_bar_ts = torch.cumprod(alpha_ts, dim=0)
        self.sqrt_alpha_bar_ts = torch.sqrt(alpha_bar_ts)
        self.sqrt_1minus_alpha_bar_ts = torch.sqrt(1.0 - alpha_bar_ts)

    def noised_sample_and_velocity_target(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_ts[t].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        sqrt_1minus_alpha_bar_t = self.sqrt_1minus_alpha_bar_ts[t].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        xt = sqrt_alpha_bar_t * x0 + sqrt_1minus_alpha_bar_t * noise
        v_target = sqrt_alpha_bar_t * noise - sqrt_1minus_alpha_bar_t * x0
        return xt, v_target


def load_dit(
    ckpt_path: str,
    device: str,
    input_h: int,
    input_w: int,
    in_channels: int,
    action_dim: int,
    max_frames: int,
    patch_size: int,
    hidden_size: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
) -> torch.nn.Module:
    if (
        input_h,
        input_w,
        in_channels,
        action_dim,
        max_frames,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
    ) == (18, 32, 16, 8, 32, 2, 1024, 16, 16, 4.0):
        model = DiT_models["DiT-S/2"]()
    else:
        model = DiT(
            input_h=input_h,
            input_w=input_w,
            in_channels=in_channels,
            external_cond_dim=action_dim,
            max_frames=max_frames,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    if ckpt_path.endswith(".pt"):
        state = torch.load(ckpt_path, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    elif ckpt_path.endswith(".safetensors"):
        load_model(model, ckpt_path)
    else:
        raise ValueError(f"Unsupported DiT checkpoint format: {ckpt_path}")

    return model.to(device).eval()


def load_vae(ckpt_path: str, device: str) -> torch.nn.Module:
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    if ckpt_path.endswith(".pt"):
        state = torch.load(ckpt_path, weights_only=True)
        vae.load_state_dict(state)
    elif ckpt_path.endswith(".safetensors"):
        load_model(vae, ckpt_path)
    else:
        raise ValueError(f"Unsupported VAE checkpoint format: {ckpt_path}")
    return vae.to(device).eval()


def get_pixel_target_size(model: torch.nn.Module) -> tuple[int, int]:
    img_size = model.x_embedder.img_size
    return int(img_size[0]), int(img_size[1])


@torch.no_grad()
def encode_frames(frames: torch.Tensor, vae: torch.nn.Module, device: str) -> torch.Tensor:
    bsz, timesteps, _, _, _ = frames.shape
    resized = resize(rearrange(frames, "b t c h w -> (b t) c h w"), VAE_TARGET_SIZE, antialias=True)
    with autocast_context(device):
        latents = vae.encode(resized * 2 - 1).mean * SCALING_FACTOR
    latents = rearrange(
        latents,
        "(b t) (h w) c -> b t c h w",
        b=bsz,
        t=timesteps,
        h=VAE_TARGET_SIZE[0] // vae.patch_size,
        w=VAE_TARGET_SIZE[1] // vae.patch_size,
    )
    return latents.float()


@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae: torch.nn.Module, device: str) -> torch.Tensor:
    bsz, timesteps, _, _, _ = latents.shape
    flattened = rearrange(latents, "b t c h w -> (b t) (h w) c")
    with autocast_context(device):
        decoded = (vae.decode(flattened / SCALING_FACTOR) + 1.0) / 2.0
    decoded = rearrange(decoded, "(b t) c h w -> b t c h w", b=bsz, t=timesteps)
    return decoded.float().clamp(0.0, 1.0)


@torch.no_grad()
def prepare_model_inputs(
    frames: torch.Tensor,
    model: torch.nn.Module,
    input_space: Literal["latent", "pixel"],
    device: str,
    vae: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        model_inputs: clip in the space the model operates in
        eval_frames: RGB frames in the resolution used for visual metrics
    """
    if input_space == "latent":
        assert vae is not None
        eval_frames = resize(
            rearrange(frames, "b t c h w -> (b t) c h w"),
            VAE_TARGET_SIZE,
            antialias=True,
        )
        eval_frames = rearrange(eval_frames, "(b t) c h w -> b t c h w", b=frames.shape[0], t=frames.shape[1])
        model_inputs = encode_frames(frames, vae, device)
        return model_inputs, eval_frames.float()

    pixel_size = get_pixel_target_size(model)
    resized = resize(
        rearrange(frames, "b t c h w -> (b t) c h w"),
        pixel_size,
        antialias=True,
    )
    resized = rearrange(resized, "(b t) c h w -> b t c h w", b=frames.shape[0], t=frames.shape[1])
    return resized.float(), resized.float()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(start_dim=1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def try_build_lpips(device: str):
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:  # pragma: no cover
        print(f"LPIPS unavailable: {exc}")
        return None


def try_build_pyiqa_metric(name: str, device: str):
    try:
        import pyiqa  # type: ignore

        return pyiqa.create_metric(name, device=device)
    except Exception as exc:  # pragma: no cover
        print(f"{name} unavailable: {exc}")
        return None


class ClipEmbedder:
    def __init__(self, device: str) -> None:
        try:
            from transformers import AutoProcessor, CLIPModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for CLIP-based temporal consistency. "
                "Install it with `pip install transformers`."
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


@dataclass
class MetricTotals:
    noise_loss_sum: float = 0.0
    noise_loss_count: int = 0
    psnr_sum: float = 0.0
    psnr_count: int = 0
    lpips_sum: float = 0.0
    lpips_count: int = 0
    musiq_sum: float = 0.0
    musiq_count: int = 0
    aesthetic_sum: float = 0.0
    aesthetic_count: int = 0
    temporal_consistency_sum: float = 0.0
    temporal_consistency_count: int = 0

    def to_dict(self) -> dict[str, float | None]:
        def safe_avg(total: float, count: int) -> float | None:
            return (total / count) if count > 0 else None

        return {
            "noise_loss": safe_avg(self.noise_loss_sum, self.noise_loss_count),
            "psnr": safe_avg(self.psnr_sum, self.psnr_count),
            "lpips": safe_avg(self.lpips_sum, self.lpips_count),
            "image_quality_musiq": safe_avg(self.musiq_sum, self.musiq_count),
            "aesthetic_quality": safe_avg(self.aesthetic_sum, self.aesthetic_count),
            "temporal_consistency": safe_avg(self.temporal_consistency_sum, self.temporal_consistency_count),
        }


@torch.no_grad()
def rollout_sequence(
    model: torch.nn.Module,
    vae: torch.nn.Module | None,
    gt_frames: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    input_space: Literal["latent", "pixel"],
    n_prompt_frames: int,
    ddim_steps: int,
    max_noise_level: int,
    noise_abs_max: float,
    stabilization_level: int,
) -> torch.Tensor:
    x0, _ = prepare_model_inputs(gt_frames, model, input_space, device, vae)
    batch_size, total_frames, _, _, _ = x0.shape

    x = x0[:, :n_prompt_frames].clone()
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1, device=device)

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = rearrange(torch.cumprod(alphas, dim=0), "t -> t 1 1 1")

    for frame_idx in range(n_prompt_frames, total_frames):
        chunk = torch.randn((batch_size, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, frame_idx + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((batch_size, frame_idx), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((batch_size, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((batch_size, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x[:, start_frame:]
            t_curr = t[:, start_frame:]
            t_next_curr = t_next[:, start_frame:]

            with autocast_context(device):
                v = model(x_curr, t_curr, actions[:, start_frame : frame_idx + 1])

            x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                1 / alphas_cumprod[t_curr] - 1
            ).sqrt()

            alpha_next = alphas_cumprod[t_next_curr]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    if input_space == "latent":
        assert vae is not None
        return decode_latents(x, vae, device)
    return x.float().clamp(0.0, 1.0)


def evaluate_noise_loss(
    model: torch.nn.Module,
    vae: torch.nn.Module | None,
    dataloader: DataLoader,
    scheduler: NoiseScheduler,
    device: str,
    input_space: Literal["latent", "pixel"],
    max_batches: int | None,
) -> tuple[float, int]:
    loss_fn = torch.nn.MSELoss(reduction="mean")
    total = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Noise loss", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        x0, _ = prepare_model_inputs(frames, model, input_space, device, vae)
        bsz, timesteps, _, _, _ = x0.shape
        t = torch.randint(0, scheduler.timesteps, (bsz, timesteps), device=device)
        noise = torch.randn_like(x0)
        xt, v_target = scheduler.noised_sample_and_velocity_target(x0, t, noise)

        with autocast_context(device):
            v_pred = model(xt, t, actions)
            loss = loss_fn(v_pred.float(), v_target.float())

        total += float(loss.item()) * bsz
        count += bsz

    return total, count


def maybe_save_rollout_video(save_dir: Path | None, sample_idx: int, pred_frames: torch.Tensor, fps: int) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    video = rearrange((pred_frames.clamp(0, 1) * 255).byte().cpu(), "t c h w -> t h w c")
    write_video(str(save_dir / f"rollout_{sample_idx}.mp4"), video, fps=fps)


def evaluate_rollouts(
    model: torch.nn.Module,
    vae: torch.nn.Module | None,
    dataset: ProcessedGameDataset,
    device: str,
    totals: MetricTotals,
    input_space: Literal["latent", "pixel"],
    n_samples: int,
    n_prompt_frames: int,
    ddim_steps: int,
    max_noise_level: int,
    noise_abs_max: float,
    stabilization_level: int,
    fps: int,
    rollout_save_dir: Path | None,
) -> None:
    lpips_metric = try_build_lpips(device)
    musiq_metric = try_build_pyiqa_metric("musiq", device)
    aesthetic_metric = try_build_pyiqa_metric("laion_aes", device)
    clip_embedder = None
    try:
        clip_embedder = ClipEmbedder(device)
    except Exception as exc:
        print(f"Temporal consistency unavailable: {exc}")

    sample_indices = torch.linspace(0, len(dataset) - 1, steps=min(n_samples, len(dataset))).long().tolist()

    for sample_num, dataset_idx in enumerate(tqdm(sample_indices, desc="Rollouts", leave=False)):
        batch = dataset[dataset_idx]
        frames = batch["frames"].unsqueeze(0).to(device)
        actions = batch["actions"].unsqueeze(0).to(device)

        pred_frames = rollout_sequence(
            model=model,
            vae=vae,
            gt_frames=frames,
            actions=actions,
            device=device,
            input_space=input_space,
            n_prompt_frames=n_prompt_frames,
            ddim_steps=ddim_steps,
            max_noise_level=max_noise_level,
            noise_abs_max=noise_abs_max,
            stabilization_level=stabilization_level,
        )

        _, gt_eval_frames = prepare_model_inputs(frames, model, input_space, device, vae)

        pred_future = pred_frames[:, n_prompt_frames:]
        gt_future = gt_eval_frames[:, n_prompt_frames:]

        psnr = compute_psnr(
            rearrange(pred_future, "b t c h w -> (b t) c h w"),
            rearrange(gt_future, "b t c h w -> (b t) c h w"),
        )
        totals.psnr_sum += float(psnr.mean().item())
        totals.psnr_count += 1

        if lpips_metric is not None:
            lpips_value = lpips_metric(
                rearrange(pred_future * 2 - 1, "b t c h w -> (b t) c h w"),
                rearrange(gt_future * 2 - 1, "b t c h w -> (b t) c h w"),
            ).mean()
            totals.lpips_sum += float(lpips_value.item())
            totals.lpips_count += 1

        if musiq_metric is not None:
            musiq_value = musiq_metric(rearrange(pred_future, "b t c h w -> (b t) c h w")).mean()
            totals.musiq_sum += float(musiq_value.item())
            totals.musiq_count += 1

        if aesthetic_metric is not None:
            aes_value = aesthetic_metric(rearrange(pred_future, "b t c h w -> (b t) c h w")).mean()
            totals.aesthetic_sum += float(aes_value.item())
            totals.aesthetic_count += 1

        if clip_embedder is not None and pred_future.shape[1] > 1:
            emb = clip_embedder.embed(rearrange(pred_future, "b t c h w -> (b t) c h w"))
            emb = rearrange(emb, "(b t) d -> b t d", b=pred_future.shape[0], t=pred_future.shape[1])
            temporal = F.cosine_similarity(emb[:, 1:], emb[:, :-1], dim=-1).mean()
            totals.temporal_consistency_sum += float(temporal.item())
            totals.temporal_consistency_count += 1

        maybe_save_rollout_video(rollout_save_dir, sample_num, pred_frames[0], fps)


def build_dataloader(data_dir: str, batch_size: int, clip_len: int, stride: int) -> DataLoader:
    dataset = ProcessedGameDataset(data_dir, clip_len=clip_len, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def main(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available() or args.device == "cpu", "CUDA is required unless you explicitly pass --device cpu"

    if args.input_space == "latent" and not args.vae_ckpt:
        raise ValueError("--vae-ckpt is required when --input-space latent")

    model = load_dit(
        ckpt_path=args.dit_ckpt,
        device=args.device,
        input_h=args.model_input_height,
        input_w=args.model_input_width,
        in_channels=args.model_in_channels,
        action_dim=args.action_dim,
        max_frames=args.max_frames,
        patch_size=args.model_patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    vae = load_vae(args.vae_ckpt, args.device) if args.input_space == "latent" else None

    dataloader = build_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        stride=args.stride,
    )
    dataset = dataloader.dataset
    assert isinstance(dataset, ProcessedGameDataset)

    totals = MetricTotals()
    scheduler = NoiseScheduler(args.max_noise_level, args.device)

    noise_total, noise_count = evaluate_noise_loss(
        model=model,
        vae=vae,
        dataloader=dataloader,
        scheduler=scheduler,
        device=args.device,
        input_space=args.input_space,
        max_batches=args.max_batches,
    )
    totals.noise_loss_sum = noise_total
    totals.noise_loss_count = noise_count

    rollout_dir = Path(args.save_dir) / "rollouts" if args.save_dir else None
    evaluate_rollouts(
        model=model,
        vae=vae,
        dataset=dataset,
        device=args.device,
        totals=totals,
        input_space=args.input_space,
        n_samples=args.num_rollout_samples,
        n_prompt_frames=args.n_prompt_frames,
        ddim_steps=args.ddim_steps,
        max_noise_level=args.max_noise_level,
        noise_abs_max=args.noise_abs_max,
        stabilization_level=args.stabilization_level,
        fps=args.fps,
        rollout_save_dir=rollout_dir,
    )

    metrics = totals.to_dict()
    print(json.dumps(metrics, indent=2))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved metrics to {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DiT checkpoint on processed SMB clips.")
    parser.add_argument("--data-dir", type=str, required=True, help="Processed dataset root created by data_utils.py.")
    parser.add_argument("--dit-ckpt", type=str, required=True, help="Path to the DiT checkpoint to evaluate.")
    parser.add_argument("--vae-ckpt", type=str, default="vit-l-20.safetensors", help="Path to the VAE checkpoint.")
    parser.add_argument(
        "--input-space",
        type=str,
        choices=["latent", "pixel"],
        default="latent",
        help="Whether the evaluated model operates in latent space or directly in RGB pixel space.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--model-input-height",
        type=int,
        default=18,
        help="Model input height in model space. Use 18 for the released latent DiT, or your pixel height for RGB models.",
    )
    parser.add_argument(
        "--model-input-width",
        type=int,
        default=32,
        help="Model input width in model space. Use 32 for the released latent DiT, or your pixel width for RGB models.",
    )
    parser.add_argument(
        "--model-in-channels",
        type=int,
        default=16,
        help="Model input channels. Use 16 for the released latent DiT or 3 for RGB pixel-space models.",
    )
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--model-patch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for noise-loss evaluation.")
    parser.add_argument("--num-rollout-samples", type=int, default=4)
    parser.add_argument("--n-prompt-frames", type=int, default=1)
    parser.add_argument("--max-noise-level", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=8)
    parser.add_argument("--noise-abs-max", type=float, default=20.0)
    parser.add_argument("--stabilization-level", type=int, default=15)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
