import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner

from mvit import MViT_V2_Lightning, FireDataModule

URL = "./data/01.원천데이터/"
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        default="train",
        help="train | valid | test | predict",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default=URL,
        nargs="+",
        help="data path",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-n", "--num_workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="number of devices. ex) 1, 2 or [0, 2]",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=False,
        action="store_true",
        help="profile model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mvit: L.LightningModule = MViT_V2_Lightning(num_classes=3)
    dm: L.LightningDataModule = FireDataModule(
        root=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    if args.profile:
        print("[INFO] Profiling Model...")
        L.Trainer(
            profiler="simple",
            max_epochs=2,
            precision="16-mixed",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        ).fit(mvit, dm)
        exit()

    trainer: L.Trainer = L.Trainer(
        logger=WandbLogger(project="mvit"),
        devices=args.devices,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[
            RichProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./checkpoints/",
                filename="mvit-{epoch:02d}-{val_acc:.2f}",
                save_top_k=5,
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )
    tuner: Tuner = Tuner(trainer)
    tuner.lr_find(mvit, dm)

    if args.stage == "train":
        trainer.fit(mvit, dm)
    elif args.stage == "valid":
        trainer.validate(mvit, dm)
    elif args.stage == "test":
        trainer.test(mvit, dm)
    elif args.stage == "predict":
        trainer.predict(mvit, dm)
