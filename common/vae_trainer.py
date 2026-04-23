from typing import cast, Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_comps.vae import AutoencoderKL

class VAETrainer:
    _MIN_LR_COE = 0.01
    _WARMUP_START_FACTOR = 0.01
    _WARMUP_END_FACTOR = 1.0
    
    def __init__(
            self,
            vae: AutoencoderKL,
            device: str,
            lr: float = 1e-4,
            kl_weight: float = 1e-6,
            grad_clip_max_norm: float = 1.0,
            warmup_steps: int = 1000,
            total_steps: int = 100000,
            debug: bool = False
    ) -> None:
        self.vae = vae
        self.device = device
        self.kl_weight = kl_weight
        self.grad_clip_max_norm = grad_clip_max_norm

        self.loss_fn = nn.MSELoss()
        
        # VAEs are usually trained with all components unfrozen
        self.freeze_report(debug)

        self.optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr)

        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=VAETrainer._WARMUP_START_FACTOR,
            end_factor=VAETrainer._WARMUP_END_FACTOR,
            total_iters=warmup_steps
        )

        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=lr * VAETrainer._MIN_LR_COE
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, 
            schedulers=[warmup_sch, cosine_sch], 
            milestones=[warmup_steps]
        )

        self.dtype = next(self.vae.parameters()).dtype

    def train_epoch(self, loader: DataLoader, n_batches_per_print: int = 500) -> tuple[float, list[float]]:
        return self._eval_train_loop(loader, n_batches_per_print, is_training=True)

    def eval_epoch(self, loader: DataLoader, n_batches_per_print: int = 500) -> tuple[float, list[float]]:
        return self._eval_train_loop(loader, n_batches_per_print, is_training=False)

    def freeze_report(self, debug: bool = False):
        trainable_params = [p for p in self.vae.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.vae.parameters())
        trainable_total = sum(p.numel() for p in trainable_params)

        print(f'\n--- VAE Freeze Report ---')
        print(f'Total Parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable Parameters: {trainable_total / 1e6:.2f}M')
        print(f'Training {100 * trainable_total / total_params:.2f}% of the VAE.')
        print(f'--------------------------\n')

    def _eval_train_loop(self, loader: DataLoader, n_print: int, is_training: bool) -> tuple[float, list[float]]:
        self.vae.train() if is_training else self.vae.eval()
        
        total_loss = 0.0
        total_samples = 0
        loss_per_step: list[float] = []

        for i, batch in enumerate(loader):
            if (i + 1) % n_print == 0 and len(loss_per_step) > 0:
                print(f'Starting batch {i + 1}/{len(loader)} (last batch loss: {loss_per_step[-1]:.6f})')

            x = batch['pixels'].to(self.device, dtype=self.dtype)
            if x.ndim == 5: # Flatten B and T if video dataset is used
                x = torch.flatten(x, start_dim=0, end_dim=1)

            loss = self._step(x, is_training)
            batch_size = x.size(0)

            total_loss += loss * batch_size
            total_samples += batch_size
            loss_per_step.append(loss)

        return total_loss / total_samples, loss_per_step

    def _step(self, x: torch.Tensor, is_training: bool) -> float:
        self.optimizer.zero_grad() if is_training else None

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            rec, post, _ = self.vae.autoencode(x)
            recon_loss = self.loss_fn(rec, x)
            
            # Standard VAE KL Divergence
            kl_loss = torch.mean(0.5 * torch.sum(
                torch.pow(post.mean, 2) + post.var - post.logvar - 1, 
                dim=[1, 2] 
            ))
            loss = recon_loss + (self.kl_weight * kl_loss)

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip_max_norm)
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()

    def get_optimizer(self) -> torch.optim.AdamW:
        return self.optimizer