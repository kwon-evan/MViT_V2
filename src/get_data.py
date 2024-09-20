import torch
import torchvision

if __name__ == "__main__":
    print(torchvision.__version__)
    kinetics = torchvision.datasets.Kinetics(
        root="./data",
        # download=True,
        num_workers=12,
        num_download_workers=12,
        frames_per_clip=16,
    )
    print(kinetics)
