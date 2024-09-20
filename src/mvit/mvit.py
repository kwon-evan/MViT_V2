from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L


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
        # freeze the pretrained model
        for param in pretrained.parameters():
            param.requires_grad = False
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
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.model = MViT_V2(num_classes)

    def forward(self, x):
        return self.model(x)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss


if __name__ == "__main__":
    model = MViT_V2(num_classes=10)

    x = torch.rand(1, 3, 16, 224, 224)
    y = model(x)
    print(y.shape, torch.argmax(y, dim=1))

    pl_model = MViT_V2_Lightning(num_classes=10)
    y = pl_model(x)
    print(y.shape, torch.argmax(y, dim=1))
