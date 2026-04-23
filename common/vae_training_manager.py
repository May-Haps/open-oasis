import os
import json
import torch
from typing import TypedDict, Any
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model_comps.vae import AutoencoderKL
from common.vae_mario_dataset import MarioPixelDataset
from common.vae_trainer import VAETrainer

class VAETrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    lr: float
    kl_weight: float
    save_dir: str


class VAETrainingResults(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    train_losses_per_step: list[list[float]]
    val_losses_per_step: list[list[float]]

class VAETrainingManager:
    _N_DATALOADER_WORKERS = 4
    _N_BATCHES_PER_PRINT = 5
    _SAVED_SAMPLES_PER_EPOCH = 4
    _WARMUP_STEP_PERCENT = 0.05

    def __init__(self, vae: AutoencoderKL, device: str = 'cuda', seed: int = 42):
        self.vae = vae.to(device)
        self.device = device
        torch.manual_seed(seed)

    def train_vae(self, train_dir: str, val_dir: str, config: dict) -> VAETrainingResults:
        train_loader = self._build_loader(train_dir, config['batch_size'], True)
        val_loader = self._build_loader(val_dir, config['batch_size'], False)

        total_steps = config['epochs'] * len(train_loader)

        trainer = VAETrainer(
            vae=self.vae, 
            device=self.device, 
            lr=config['lr'], 
            kl_weight=config['kl_weight'],
            warmup_steps=total_steps * VAETrainingManager._WARMUP_STEP_PERCENT,
            total_steps=total_steps
        )

        return self._run_training(trainer, train_loader, val_loader, config)

    def _run_training(self, trainer, train_loader, val_loader, config) -> VAETrainingResults:
        results: VAETrainingResults = {
            'train_losses': [], 'val_losses': [],
            'train_losses_per_step': [], 'val_losses_per_step': []
        }

        print(f'VAE Training batches: {len(train_loader)}')
        print(f'VAE Val batches: {len(val_loader)}')

        for epoch in range(1, config['epochs'] + 1):
            print(f'\n------------------------------ VAE Epoch {epoch}/{config["epochs"]} ------------------------------')
            
            print(f'Starting training epoch')
            t_loss, t_steps = trainer.train_epoch(train_loader, self._N_BATCHES_PER_PRINT)
            
            print(f'Starting validation epoch')
            v_loss, v_steps = trainer.eval_epoch(val_loader, self._N_BATCHES_PER_PRINT)

            results['train_losses'].append(t_loss)
            results['val_losses'].append(v_loss)
            results['train_losses_per_step'].append(t_steps)
            results['val_losses_per_step'].append(v_steps)

            print(f'Epoch {epoch} complete - Train Loss: {t_loss:.6f}, Val Loss: {v_loss:.6f}')

            # Visual check
            self._save_samples(val_loader, epoch, config['save_dir'])
            
            # Checkpoint
            self._save_checkpoint(epoch, config['save_dir'])

        self._save_results(results, config['save_dir'])
        return results

    def _save_samples(self, loader, epoch, save_dir):
        self.vae.eval()
        batch = next(iter(loader))
        x = batch['pixels'][:VAETrainingManager._SAVED_SAMPLES_PER_EPOCH].to(self.device)
        if x.ndim == 5: x = x[:, 0] # Take first frame if video
        
        with torch.no_grad():
            rec, _, _ = self.vae.autoencode(x)
        
        comparison = torch.cat([x, rec], dim=0) 
        path = os.path.join(save_dir, f"recon_ep{epoch}.png")
        save_image(comparison, path, nrow=4, normalize=True)
        print(f'Generated reconstruction sample: {path}')

    def _save_checkpoint(self, epoch, save_dir):
        path = os.path.join(save_dir, f"vae_ckpt_ep{epoch}.pt")
        torch.save(self.vae.state_dict(), path)
        print(f'Saved VAE checkpoint: {path}')

    def _save_results(self, results, save_dir):
        path = os.path.join(save_dir, "vae_training_results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved VAE training results to {path}')

    def _build_loader(self, path, batch_size, shuffle):
        dataset = MarioPixelDataset(path)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=self._N_DATALOADER_WORKERS, pin_memory=True
        )