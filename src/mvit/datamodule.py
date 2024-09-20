import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import Kinetics


class KineticDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train = Kinetics(
                self.data_dir,
                split="train",
                frames_per_clip=16,
                num_workers=self.num_workers,
                num_download_workers=self.num_workers,
            )
            self.val = Kinetics(
                self.data_dir,
                split="val",
                frames_per_clip=16,
                num_workers=self.num_workers,
                num_download_workers=self.num_workers,
            )
        if stage == "test":
            self.test = Kinetics(
                self.data_dir,
                split="test",
                frames_per_clip=16,
                num_workers=self.num_workers,
                num_download_workers=self.num_workers,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = KineticDataModule()
    dm.setup()
