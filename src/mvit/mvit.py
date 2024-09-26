from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import Accuracy, F1Score


def _unsqueeze(
    x: torch.Tensor, target_dim: int, expand_dim: int
) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


class MViT_V2(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        weight = torchvision.models.video.MViT_V2_S_Weights.DEFAULT
        pretrained = torchvision.models.video.mvit_v2_s(weights=weight)
        output_dim = 768

        # get the pretrained model's layers
        self.conv_proj = pretrained.conv_proj
        self.pos_encoding = pretrained.pos_encoding
        self.blocks = pretrained.blocks
        self.norm = pretrained.norm
        # initialize head with new weights
        self.head = nn.Sequential(
            nn.Dropout(0.5, inplace=True),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return x


class MViT_V2_Lightning(L.LightningModule):
    def __init__(self, num_classes: int = 1000, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

        self.model = MViT_V2(num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        logits = self.model(x)
        loss = self.loss(logits, label)

        label_idx = torch.argmax(label, dim=1)

        acc = self.acc(logits, label_idx)
        f1 = self.f1(logits, label_idx)

        self.__log__(stage="train", loss=loss, acc=acc, f1=f1)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        logits = self.model(x)
        loss = self.loss(logits, label)

        label_idx = torch.argmax(label, dim=1)

        acc = self.acc(logits, label_idx)
        f1 = self.f1(logits, label_idx)

        self.__log__(stage="val", loss=loss, acc=acc, f1=f1)
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        logits = self.model(x)
        loss = self.loss(logits, label)

        label_idx = torch.argmax(label, dim=1)

        acc = self.acc(logits, label_idx)
        f1 = self.f1(logits, label_idx)

        self.__log__(stage="test", loss=loss, acc=acc, f1=f1)
        return loss

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-5,
            max_lr=1e-3,
            gamma=0.85,
            mode="exp_range",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def __log__(self, stage: str, **kwargs: dict[str, torch.Tensor]) -> None:
        self.log_dict(
            {f"{stage}_{k}": v for k, v in kwargs.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


if __name__ == "__main__":
    model = MViT_V2(num_classes=10)

    x = torch.rand(1, 3, 16, 224, 224)
    y = model(x)
    print(y.shape, torch.argmax(y, dim=1))

    pl_model = MViT_V2_Lightning(num_classes=10)
    y = pl_model(x)
    print(y.shape, torch.argmax(y, dim=1))
