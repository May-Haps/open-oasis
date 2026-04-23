from typing import Any, cast, TypedDict
import json
import os

import torch
from torch.utils.data import DataLoader

from model.dit import DiT
from data.dataset import MarioPixelDataset
from training.model_trainer import ModelTrainer
from training.rollout_sampler import RolloutSampler


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
    trainable_components: list[str]
    save_dir: str | None


class ModelTrainingResults(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    train_losses_per_step: list[list[float]]
    val_losses_per_step: list[list[float]]


class CheckpointState(TypedDict):
    epoch: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


class TrainingManager():
    _N_DATALOADER_WORKERS = 4
    _N_BATCHES_PER_PRINT = 500
    _CKPT_FILE_PREFIX = 'ckpt'
    _TRAINING_RESULTS_FILENAME = 'training_result.json'
    _N_ROLLOUT_SAMPLES = 4
    _N_ROLLOUT_FRAMES = 24
    _N_ROLLOUT_DDIM_STEPS = 8

    def __init__(
            self,
            dit: DiT,
            device: str = 'cuda',
            seed: int = 42,
            debug: bool = False
    ) -> None:
        self.dit = dit.to(device)
        self.device = device
        self.debug = debug
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

        train_loader = self._build_dataloader(
            train_dir, config['batch_size'], config['clip_len'], config['clip_stride'], shuffle=True
        )
        val_loader = self._build_dataloader(
            val_dir, config['batch_size'], config['clip_len'], config['clip_stride'], shuffle=False
        )

        rollout_sampler = None
        if config['save_dir'] is not None:
            rollout_sampler = RolloutSampler(
                dit=self.dit,
                val_dataset=cast(MarioPixelDataset, val_loader.dataset),
                device=self.device,
                num_samples=TrainingManager._N_ROLLOUT_SAMPLES,
                num_frames=TrainingManager._N_ROLLOUT_FRAMES,
                ddim_steps=TrainingManager._N_ROLLOUT_DDIM_STEPS,
            )

        trainer = ModelTrainer(
            dit=self.dit,
            max_noise_level=config['max_noise_level'],
            device=self.device,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            grad_clip_max_norm=config['grad_clip_max_norm'],
            trainable_components=config['trainable_components'],
            debug=self.debug,
        )

        return self._run_training(trainer, train_loader, val_loader, rollout_sampler, config)

    def _run_training(
            self,
            trainer: ModelTrainer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            rollout_sampler: RolloutSampler | None,
            config: ModelTrainingConfig,
    ) -> ModelTrainingResults:
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_losses_per_step: list[list[float]] = []
        val_losses_per_step: list[list[float]] = []

        print(f'Training batches: {len(train_loader)}')
        print(f'Val batches: {len(val_loader)}')

        for epoch in range(1, config['epochs'] + 1):
            print(f'------------------------------ Epoch {epoch}/{config["epochs"]} ------------------------------')
            train_loss, epoch_train_steps = trainer.train_epoch(train_loader, TrainingManager._N_BATCHES_PER_PRINT)
            val_loss, epoch_val_steps = trainer.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_losses_per_step.append(epoch_train_steps)
            val_losses_per_step.append(epoch_val_steps)

            print(f'Epoch {epoch} - train loss: {train_loss:.5f}, val loss: {val_loss:.5f}')

            if rollout_sampler is not None and config['save_dir'] is not None:
                print('Generating rollout samples')
                for p in rollout_sampler.sample(epoch, config['save_dir']):
                    print(f'\tsaved {p}')

            if config['save_dir'] is not None:
                self._save_checkpoint(trainer, config['save_dir'], epoch)

        results = ModelTrainingResults(
            train_losses=train_losses,
            val_losses=val_losses,
            train_losses_per_step=train_losses_per_step,
            val_losses_per_step=val_losses_per_step,
        )

        if config['save_dir'] is not None:
            self._save_training_results(results, config['save_dir'])

        return results

    def _build_dataloader(
            self,
            data_folder_path: str,
            batch_size: int,
            clip_len: int,
            clip_stride: int,
            shuffle: bool,
    ) -> DataLoader:
        dataset = MarioPixelDataset(data_folder_path, clip_len, clip_stride)
        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=TrainingManager._N_DATALOADER_WORKERS,
            pin_memory=TrainingManager._N_DATALOADER_WORKERS > 0,
        )

    def _save_checkpoint(self, trainer: ModelTrainer, save_dir: str, epoch: int) -> None:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f'{TrainingManager._CKPT_FILE_PREFIX}{epoch}.pt'
        torch.save(CheckpointState(
            epoch=epoch,
            model=self.dit.state_dict(),
            optimizer=trainer.get_optimizer().state_dict(),
            scheduler=trainer.get_scheduler().state_dict(),
        ), os.path.join(save_dir, ckpt_name))

    def _save_training_results(self, results: ModelTrainingResults, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, TrainingManager._TRAINING_RESULTS_FILENAME)
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)
