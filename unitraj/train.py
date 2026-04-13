import datetime

import lightning.pytorch as pl
import torch

torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    exp_name = f"{cfg.model_name}_{timestamp}"
    output_path = f"./ckpt/{exp_name}"

    if cfg.method.model_name == 'VBD':

        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path is not None:
            print("Load Weights from ", ckpt_path)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])

        if not cfg.get("train_encoder", True):
            encoder_path = cfg.get("encoder_ckpt", None)
            if encoder_path is not None:
                model_dict = torch.load(encoder_path, map_location="cpu")["state_dict"]
                model_dict = {k: v for k, v in model_dict.items() if k.startswith("encoder.")}
                model.load_state_dict(model_dict, strict=False)
                print("Load Encoder Weights")
            else:
                cfg.train_encoder = True
                import warnings
                warnings.warn("Encoder path is not provided, will train encoder from scratch")

        call_backs = [
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=20,
                monitor="val/loss",
                filename="epoch={epoch:02d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
    else:
        call_backs = [
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=1,
                monitor='val/brier_fde',  # Replace with your validation metric
                filename='{epoch}-{val/brier_fde:.2f}',
                mode='min',  # 'min' for loss/error, 'max' for accuracy
            )
        ]

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False, shuffle=True,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    use_wandb = cfg.get("use_wandb", True)
    logger = (WandbLogger(name=exp_name, project=cfg.project, entity=cfg.username, dir=output_path)
              if use_wandb else CSVLogger(output_path, name=exp_name, version=1))

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=logger,
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj', exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.get('init_from'))


if __name__ == '__main__':
    train()
