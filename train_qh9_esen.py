
import argparse
import os
import sys
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone

# Ensure local imports work when running as a script.
sys.path.append(os.path.dirname(__file__))
from qh9_dataset import QH9AtomicDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESEN on QH9 splits.")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="/root/25DFT/QHFlow/dataset/QH9Stable_shard/processed",
        help="Path to the processed QH9 dataset directory.",
    )
    parser.add_argument(
        "--split_filename",
        type=str,
        default="processed_QH9Stable_random_12.json",
        help="JSON file that holds train/val/test splits.",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=2048)
    parser.add_argument("--test_limit", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--force_weight", type=float, default=100.0)
    parser.add_argument("--output_dir", type=str, default="qh9_esen_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--always_use_pbc",
        action="store_true",
        help="Force eSCN to assume periodic boundary conditions.",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=80,
        help="Maximum neighbors used for on-the-fly graph construction.",
    )
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--monitor_metric", type=str, default="val/loss")
    parser.add_argument("--monitor_mode", type=str, default="min")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default="qh9-esen")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="qh9-esen-run")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--dataset_debug", action="store_true")
    parser.add_argument("--dataset_debug_interval", type=int, default=1000)
    return parser.parse_args()


def create_model(args: argparse.Namespace) -> HydraModelV2:
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=128,
        hidden_channels=128,
        edge_channels=128,
        num_layers=4,
        lmax=2,
        mmax=1,
        otf_graph=True,
        cutoff=5.0,
        max_neighbors=args.max_neighbors,
        regress_forces=True,
        direct_forces=False,
        always_use_pbc=args.always_use_pbc,
        use_dataset_embedding=False,
    )
    head = MLP_EFS_Head(backbone)
    return HydraModelV2(backbone, {"energy_forces": head})


class QH9DataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_dataset(self, split_name: Optional[str], limit: Optional[int]):
        if split_name is None:
            return None
        dataset = QH9AtomicDataset(
            processed_dir=self.args.processed_dir,
            split=split_name,
            split_filename=self.args.split_filename,
            max_samples=limit,
            debug=self.args.dataset_debug,
            debug_interval=self.args.dataset_debug_interval,
        )
        if len(dataset) == 0:
            return None
        print(
            f"[DataModule] Loaded split '{split_name}' "
            f"with {len(dataset)} samples (limit={limit})"
        )
        return dataset

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] setup(stage={stage})")
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(
                self.args.train_split, self.args.train_limit
            )
            self.val_dataset = self._build_dataset(
                self.args.val_split, self.args.val_limit
            )
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset(
                self.args.test_split, self.args.test_limit
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Training dataset is not available.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=atomicdata_list_to_batch,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=atomicdata_list_to_batch,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=atomicdata_list_to_batch,
        )


class ESENLightningModule(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.force_weight = args.force_weight
        self.lr = args.lr
        self.model = create_model(args)

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, stage: str):
        batch = batch.to(self.device)
        outputs = self(batch)
        preds = outputs["energy_forces"]
        pred_energy = preds["energy"]["energy"].squeeze()
        pred_forces = preds["forces"]["forces"]
        target_energy = batch.energy.squeeze()
        target_forces = batch.forces

        loss_energy = F.mse_loss(pred_energy, target_energy)
        loss_forces = F.mse_loss(pred_forces, target_forces)
        loss = loss_energy + self.force_weight * loss_forces

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch.natoms.shape[0],
        )
        self.log(
            f"{stage}/energy",
            loss_energy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            batch_size=batch.natoms.shape[0],
        )
        self.log(
            f"{stage}/forces",
            loss_forces,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            batch_size=batch.natoms.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def build_logger(args: argparse.Namespace):
    if args.disable_wandb:
        return None
    os.makedirs(args.output_dir, exist_ok=True)
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        save_dir=args.output_dir,
    )
    logger.log_hyperparams(vars(args))
    return logger


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    import pdb; pdb.set_trace()
    print("[Main] Creating model...")
    model = ESENLightningModule(args)
    print("[Main] Model created.")

    print("[Main] Creating data module...")
    data_module = QH9DataModule(args)
    print("[Main] Data module created.")

    print("[Main] Initializing checkpoints, loggers, and trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="qh9-esen-{epoch:03d}",
        save_top_k=args.save_top_k,
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = build_logger(args)

    progress_bar = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=max(args.grad_clip, 0.0),
        precision=args.precision,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )

    print("[Main] Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("[Main] Training finished. Running test set...")
    data_module.setup("test")
    if data_module.test_dataset is not None:
        trainer.test(model, datamodule=data_module, ckpt_path="best")
        print("[Main] Test run complete.")


if __name__ == "__main__":
    main()