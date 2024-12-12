from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import glob
import os
from torchvision.transforms import transforms
import torchvision


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label, path)
    


class BP_Eye_Dataset(Dataset):
    def __init__(self, image_paths, classes, my_transforms):
        self.image_paths = image_paths
        self.classes = classes
        self.my_transforms = my_transforms

        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx] 
        label = image_filepath.split('/')[-2]

        image = torchvision.io.read_image(image_filepath)
        
        image = self.my_transforms(image)
        label_final = self.class_to_idx[label]

        return image, label_final


my_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((256,256))
])
    

class GlaucomaEyepacsModule(LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int=0, pin_memory: bool=False, root_dir:str="/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_separated_train_test_val/", classes:list=['NRG', 'RG']) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size
        self.root_dir = root_dir
        self.classes = classes

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            #TODO CHANGE THIS WHEN DOING FULL PIPELINE
            train_data_paths = glob.glob(self.root_dir+"train/*/*.jpg")[:5]
            val_data_paths = glob.glob(self.root_dir+"val/*/*.jpg")[:5]
            test_data_paths = glob.glob(self.root_dir+"test/*/*.jpg")[:5]

            self.data_train = BP_Eye_Dataset(train_data_paths, classes=self.classes, my_transforms=my_transforms)
            self.data_val = BP_Eye_Dataset(val_data_paths, classes=self.classes, my_transforms=my_transforms)
            self.data_test = BP_Eye_Dataset(test_data_paths, classes=self.classes, my_transforms=my_transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,)

if __name__ == "__main__":
    _ = GlaucomaEyepacsModule()