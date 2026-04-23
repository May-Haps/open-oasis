from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dit import DiT
from training.noise_scheduler import NoiseScheduler

class ModelTrainer():
    def __init__(
            self,
            dit: DiT,
            max_noise_level: int,
            device: str,
            lr: float = 1e-4,
            weight_decay:float = 0.01,
            warmup_steps: int = 1000,
            grad_clip_max_norm: float = 1.0,
            trainable_components: list[str] = [],
            debug: bool = False
    ) -> None:
        self.dit = dit
        self.max_noise_level = max_noise_level
        self.device = device
        self.grad_clip_max_norm = grad_clip_max_norm

        self.input_h, self.input_w = dit.x_embedder.img_size
        self.input_c = dit.in_channels
        self.dit_action_dim = dit.external_cond_dim

        self.noise_scheduler = NoiseScheduler(max_noise_level, device)
        self.loss_fn = nn.MSELoss()
        
        self.freeze_model_components(trainable_components, debug)

        # NOTE Need to freeze specific parameters also maybe should do parameters groups
        self.optimizer = torch.optim.AdamW(self.dit.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            1.0 / warmup_steps,
            total_iters=warmup_steps
        )

        self.dtype = next(self.dit.parameters()).dtype

    def train_epoch(self, dataset_loader: DataLoader, n_batches_per_print: int = 1000) -> tuple[float, list[float]]:
        return self._eval_train_loop(dataset_loader, n_batches_per_print, is_training=True)

    def eval_epoch(self, dataset_loader: DataLoader, n_batches_per_print: int = 1000) -> tuple[float, list[float]]:
        return self._eval_train_loop(dataset_loader, n_batches_per_print, is_training=False)

    def freeze_model_components(
            self,
            trainable_components: list[str],
            debug: bool = False
    ) -> list[nn.Parameter]:
        if not trainable_components:
            print('Training all parameters (no freeze)')
            total_params = sum(p.numel() for p in self.dit.parameters())
            print(f'\n--- Freeze Report ---')
            print(f'Total Parameters: {total_params / 1e6:.2f}M')
            print(f'Trainable Parameters: {total_params / 1e6:.2f}M')
            print(f'Fine-tuning 100.00% of the model.')
            print(f'----------------------\n')
            return list(self.dit.parameters())

        print(f'Training restricted to keywords: {", ".join(trainable_components)}')
        for param in self.dit.parameters():
            param.requires_grad = False

        trainable_params: list[nn.Parameter] = []
        for name, param in self.dit.named_parameters():
            if any(key in name for key in trainable_components):
                param.requires_grad = True
                trainable_params.append(param)
                if debug:
                    print(f'Unfrozen: {name}')
            elif debug:
                print(f'Frozen: {name}')

        total_params = sum(p.numel() for p in self.dit.parameters())
        trainable_total = sum(p.numel() for p in trainable_params)
        
        assert total_params > 0

        print(f'\n--- Freeze Report ---')
        print(f'Total Parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable Parameters: {trainable_total / 1e6:.2f}M')
        print(f'Fine-tuning {100 * trainable_total / total_params:.2f}% of the model.')
        print(f'----------------------\n')
    
        return trainable_params

    def get_optimizer(self) -> torch.optim.AdamW:
        return self.optimizer

    def get_scheduler(self) -> torch.optim.lr_scheduler.LinearLR:
        return self.lr_scheduler

    def _eval_train_loop(
            self,
            dataset_loader: DataLoader,
            n_batches_per_print: int,
            is_training: bool
    ) -> tuple[float, list[float]]:
        if is_training:
            self.dit.train()
            epoch_step_fn = self._train_step
        else:
            self.dit.eval()
            epoch_step_fn = self._eval_step

        n_batches = len(dataset_loader)
        total_loss = 0.0
        total_samples = 0

        batch: dict[str, torch.Tensor]

        loss_per_step: list[float] = []

        for i, batch in enumerate(dataset_loader):
            if ((i + 1) % n_batches_per_print == 0):
                print(f'Starting batch {i + 1}/{n_batches} (last batch loss: {loss_per_step[-1]:5f})')

            x0 = batch['frames'].to(self.device, dtype=self.dtype)
            actions = batch['actions'].to(self.device, dtype=self.dtype)
            self._validate_inputs(x0, actions)

            loss = epoch_step_fn(x0, actions)
            batch_size = x0.size(dim=0)

            total_loss += loss * batch_size
            total_samples += batch_size

            loss_per_step.append(loss)
        
        return total_loss / total_samples, loss_per_step

    def _train_step(self, x0: torch.Tensor, actions: torch.Tensor | None) -> float:
        B, T, _, _, _ = x0.size()

        self.optimizer.zero_grad()

        t = torch.randint(0, self.max_noise_level, (B,T), device=self.device)
        noise = torch.randn_like(x0)
        
        xt, v_target = self.noise_scheduler.noised_sample_and_velocity_target(x0, t, noise)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_pred = self.dit(xt, t, actions)
            loss = cast(torch.Tensor, self.loss_fn(v_pred, v_target))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_norm=self.grad_clip_max_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()
    
    @torch.no_grad()
    def _eval_step(self, x0: torch.Tensor, actions: torch.Tensor | None) -> float:
        B, T, _, _, _ = x0.size()

        t = torch.randint(0, self.max_noise_level, (B,T), device=self.device)
        noise = torch.randn_like(x0)

        xt, v_target = self.noise_scheduler.noised_sample_and_velocity_target(x0, t, noise)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_pred = self.dit(xt, t, actions)
            loss = cast(torch.Tensor, self.loss_fn(v_pred, v_target))

        return loss.item()

    def _validate_inputs(self, x0, actions) -> None:
        B, T, C, H, W = x0.size()
        
        assert C == self.input_c, f'Expected C = {self.input_c} but got {C}'
        assert H == self.input_h, f'Expected H = {self.input_h} but got {H}'
        assert W == self.input_w, f'Expected W = {self.input_w} but got {W}'

        if actions is not None:
            Ba, Ta, action_dim = actions.size()
            assert B == Ba and T == Ta and self.dit_action_dim == action_dim, \
                f'Actions not None => expected (Ba, Ta, action_dim) = {(B, T, self.dit_action_dim)} but got {(Ba, Ta, action_dim)}'