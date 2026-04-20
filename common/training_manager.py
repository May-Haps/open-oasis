from typing import Any, cast, TypedDict
from datetime import datetime
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dit import DiT
from dataset import MinecraftLatentDataset
from common.model_trainer import ModelTrainer
from common.rollout_sampler import RolloutSampler
from vae import AutoencoderKL

class ModelTrainingConfig(TypedDict):
    max_noise_level: int
    clip_len: int
    clip_stride: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip_max_norm: float
    save_dir: str | None

# TODO
# class ModelEvaluationMetrics(TypedDict):
#   psnr: float
#   lpips: float
#   image_quality: float
#   aesthetic_quality: float
#   temporal_consistency: float

# TODO Should we save optimizer state?

class ModelTrainingResults(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    #val_metrics: list[ModelEvaluationMetrics]

class CheckpointState(TypedDict):
    epoch: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]

class TrainingManager():
    _N_DATALOADER_WORKERS = 4
    _N_BATCHES_PER_PRINT = 1000
    _CKPT_FILE_PREFIX = 'ckpt'
    _TRAINING_RESULTS_FILENAME = 'training_result.json'
    _N_ROLLOUT_SAMPLES = 4
    _N_ROLLOUT_FRAMES = 32
    _N_ROLLOUT_DDIM_STEPS = 8

    def __init__(self, dit: DiT, vae: AutoencoderKL | None, device: str = 'cuda', seed: int = 42) -> None:
        self.dit = dit.to(device)
        self.vae = vae.to(device) if vae is not None else None
        self.device = device
        torch.manual_seed(seed)

    def train_model(self, train_dir: str, val_dir: str, config: ModelTrainingConfig) -> ModelTrainingResults:
        assert config['max_noise_level'] > 0 
        assert config['clip_len'] > 0
        assert config['clip_stride'] > 0
        assert config['epochs'] > 0
        assert config['batch_size'] > 0
        assert config['lr'] > 0
        assert config['weight_decay'] >= 0
        assert config['warmup_steps'] > 0
        assert config['grad_clip_max_norm'] > 0
        assert self.vae is not None and config['save_dir'] is not None

        train_loader = self._build_dataloader(
            train_dir,
            config['batch_size'],
            config['clip_len'],
            config['clip_stride'],
            shuffle=True
        )
        val_loader = self._build_dataloader(
            val_dir,
            config['batch_size'],
            config['clip_len'],
            config['clip_stride'],
            shuffle=False
        )

        if config['save_dir'] is not None:
            rollout_sampler = RolloutSampler(
                dit=self.dit,
                vae=self.vae,
                val_dataset=cast(MinecraftLatentDataset, val_loader.dataset),
                device=self.device,
                num_samples=TrainingManager._N_ROLLOUT_SAMPLES,
                num_frames=TrainingManager._N_ROLLOUT_FRAMES,
                ddim_steps=TrainingManager._N_ROLLOUT_DDIM_STEPS,
            )
        else:
            rollout_sampler = None

        trainer = ModelTrainer(
            dit=self.dit,
            max_noise_level=config['max_noise_level'],
            device=self.device,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            grad_clip_max_norm=config['grad_clip_max_norm']
        )

        results = self._run_training(trainer, train_loader, val_loader, rollout_sampler, config)

        return results
    
    def _run_training(
            self,
            trainer: ModelTrainer,
            train_loader: DataLoader[MinecraftLatentDataset],
            val_loader: DataLoader[MinecraftLatentDataset],
            rollout_sampler: RolloutSampler | None,
            config: ModelTrainingConfig
    ) -> ModelTrainingResults:
        train_losses: list[float] = []
        val_losses: list[float] = []
        # val_metrics: list[ModelEvaluationMetrics] = []

        for epoch in range(1, config['epochs'] + 1):
            print(f'------------------------------ Epoch {epoch}/{config['epochs']} ------------------------------')
            print(f'Starting training epoch')
            train_loss = trainer.train_epoch(train_loader, TrainingManager._N_BATCHES_PER_PRINT)

            print(f'Starting validation epoch')
            val_loss = trainer.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch} - train loss: {train_loss:.5f}, val loss: {val_loss:.5f}')

            self._compute_evaluation_metrics(val_loader, rollout_sampler, epoch, config['save_dir'])
            # metrics = self._compute_evaluation_metrics(val_loader, rollout_sampler, epoch, config['save_dir'])
            # val_metrics.append(metrics)
            # self._print_evaluation_metrics(metrics)

            if config['save_dir'] is not None:
                self._save_checkpoint(trainer, config['save_dir'], epoch)

        results = self._format_results(train_losses, val_losses)
        # results = self._format_results(train_losses, val_losses, val_metrics)

        if config['save_dir'] is not None:
            self._save_training_results(results, config['save_dir'])

        return results

    def _compute_evaluation_metrics(
            self,
            val_loader: DataLoader[MinecraftLatentDataset],
            rollout_sampler: RolloutSampler | None,
            epoch: int,
            save_dir: str | None
    ) -> None:
        if save_dir is not None:
            assert rollout_sampler is not None
            print('Generating rollout samples')
            paths = rollout_sampler.sample(epoch, save_dir)
            for p in paths:
                print(f'\tsaved {p}')

    def _build_dataloader(
            self,
            data_folder_path: str,
            batch_size: int,
            clip_len: int,
            clip_stride: int,
            shuffle: bool
    ) -> DataLoader[MinecraftLatentDataset]:
        dataset = MinecraftLatentDataset(data_folder_path, clip_len, clip_stride)
        dataset_loader = DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=TrainingManager._N_DATALOADER_WORKERS,
            pin_memory=TrainingManager._N_DATALOADER_WORKERS > 0
        )
        return dataset_loader

    def _format_results(
            self,
            train_losses: list[float],
            val_losses: list[float],
            # val_metrics: list[ModelEvaluationMetrics]
    ) -> ModelTrainingResults:
        return ModelTrainingResults({
            'train_losses': train_losses,
            'val_losses': val_losses,
            # 'val_metrics': val_metrics
        })

    def _save_checkpoint(self, trainer: ModelTrainer, save_dir: str, epoch: int) -> None:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f'{TrainingManager._CKPT_FILE_PREFIX}{epoch}.pt'
        full_path = os.path.join(save_dir, ckpt_name)
        torch.save(CheckpointState(
            epoch=epoch,
            model=self.dit.state_dict(),
            optimizer=trainer.get_optimizer().state_dict(),
            scheduler=trainer.get_scheduler().state_dict()
        ), full_path)

    def _save_training_results(self, results: ModelTrainingResults, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, TrainingManager._TRAINING_RESULTS_FILENAME)
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)
        