from __future__ import annotations
from pathlib import Path

import torch
from einops import rearrange
from torch import autocast
from torchvision.io import write_video

from model.dit import DiT
from model.utils import sigmoid_beta_schedule
from data.dataset import MarioPixelDataset


class RolloutSampler:
    """
    Generates short autoregressive rollouts from val prompts after each epoch.
    Operates directly in pixel space — no VAE needed.
    """

    def __init__(
        self,
        dit: DiT,
        val_dataset: MarioPixelDataset,
        device: str,
        num_samples: int = 2,
        num_frames: int = 16,
        n_prompt_frames: int = 1,
        ddim_steps: int = 8,
        max_noise_level: int = 1000,
        noise_abs_max: float = 20.0,
        stabilization_level: int = 15,
        fps: int = 30,
        seed: int = 0,
    ) -> None:
        assert num_frames <= val_dataset.clip_len
        assert n_prompt_frames < num_frames

        self.dit = dit
        self.device = device
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.n_prompt_frames = n_prompt_frames
        self.ddim_steps = ddim_steps
        self.max_noise_level = max_noise_level
        self.noise_abs_max = noise_abs_max
        self.stabilization_level = stabilization_level
        self.fps = fps
        self.seed = seed

        total = len(val_dataset)
        step = max(1, total // num_samples)
        self.prompt_indices = [min(i * step, total - 1) for i in range(num_samples)]
        self.val_dataset = val_dataset

        betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = rearrange(torch.cumprod(alphas, dim=0), "T -> T 1 1 1")

    @torch.no_grad()
    def sample(self, epoch: int, save_dir: str) -> list[str]:
        out_dir = Path(save_dir) / "rollouts" / f"epoch_{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)

        was_training = self.dit.training
        self.dit.eval()

        saved_paths: list[str] = []
        for i, prompt_idx in enumerate(self.prompt_indices):
            batch = self.val_dataset[prompt_idx]
            # frames are [0,1] float; convert to [-1,1] for diffusion
            frames = batch["frames"][: self.num_frames].unsqueeze(0).to(self.device) * 2 - 1
            actions = batch["actions"][: self.num_frames].unsqueeze(0).to(self.device)

            video = self._rollout(frames, actions)
            path = out_dir / f"sample_{i}_ep{batch['episode_id']}_s{batch['start']}.mp4"
            write_video(str(path), video, fps=self.fps)
            saved_paths.append(str(path))

        if was_training:
            self.dit.train()
        return saved_paths

    def _rollout(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        B = frames.shape[0]
        total_frames = frames.shape[1]
        x = frames[:, : self.n_prompt_frames].clone()

        noise_range = torch.linspace(-1, self.max_noise_level - 1, self.ddim_steps + 1)

        for i in range(self.n_prompt_frames, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=self.device, generator=gen)
            chunk = torch.clamp(chunk, -self.noise_abs_max, self.noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - self.dit.max_frames)

            for noise_idx in reversed(range(1, self.ddim_steps + 1)):
                t_ctx = torch.full((B, i), self.stabilization_level - 1, dtype=torch.long, device=self.device)
                t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=self.device)
                t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x[:, start_frame:]
                t_curr = t[:, start_frame:]
                t_next_curr = t_next[:, start_frame:]

                with autocast("cuda", dtype=torch.bfloat16):
                    v = self.dit(x_curr, t_curr, actions[:, start_frame : i + 1])

                x_start = self.alphas_cumprod[t_curr].sqrt() * x_curr - (1 - self.alphas_cumprod[t_curr]).sqrt() * v
                x_noise = ((1 / self.alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                    1 / self.alphas_cumprod[t_curr] - 1
                ).sqrt()

                alpha_next = self.alphas_cumprod[t_next_curr]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        # [-1,1] → [0,255] uint8
        pixels = (x.clamp(-1, 1) + 1) / 2
        pixels = rearrange(pixels, "b t c h w -> b t h w c")
        pixels = (pixels * 255).byte().cpu()
        return pixels[0]
