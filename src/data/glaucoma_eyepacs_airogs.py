from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import glob
import os
from torchvision.transforms import transforms
import torchvision
import pandas as pd 



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

        # print(label) # RG->1 , NRG->0
        # print(label_final)

        return image, label_final


my_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.ToPILImage(), 
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(size=(224,224),antialias=True)
])
    

class GlaucomaEyepacsModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, root_dir:str, classes:list) -> None:
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
            # train_data_paths = sorted(glob.glob(self.root_dir+"train/*/*.jpg"))
            # val_data_paths = sorted(glob.glob(self.root_dir+"val/*/*.jpg"))
            # test_data_paths = sorted(glob.glob(self.root_dir+"test/*/*.jpg"))

            # print(train_data_paths)
            # print(val_data_paths)
            # print(test_data_paths)

            #TODO CHANGE THIS WHEN DOING FULL PIPELINE
            # train_data_paths = [glob.glob(self.root_dir+"train/RG/*.jpg")[0], glob.glob(self.root_dir+"train/NRG/*.jpg")[0]]
            # val_data_paths = [glob.glob(self.root_dir+"val/RG/*.jpg")[0], glob.glob(self.root_dir+"val/NRG/*.jpg")[0]]
            # test_data_paths = [glob.glob(self.root_dir+"test/RG/*.jpg")[0], glob.glob(self.root_dir+"test/NRG/*.jpg")[0]]

            # train_data_paths = sorted(glob.glob(self.root_dir+"train/RG/*.jpg"))[:2] + sorted(glob.glob(self.root_dir+"train/NRG/*.jpg"))[:196]
            # val_data_paths = sorted(glob.glob(self.root_dir+"val/RG/*.jpg"))[2:102] + sorted(glob.glob(self.root_dir+"val/NRG/*.jpg"))[196:296]
            # test_data_paths = sorted(glob.glob(self.root_dir+"test/RG/*.jpg"))[102:202] + sorted(glob.glob(self.root_dir+"test/NRG/*.jpg"))[296:396]

            # print(len(sorted(glob.glob(self.root_dir+"train/RG/*.jpg")))) # 2616
            # print(len(sorted(glob.glob(self.root_dir+"val/RG/*.jpg")))) # 327
            # print(len(sorted(glob.glob(self.root_dir+"test/RG/*.jpg")))) # 327

            # print(len(sorted(glob.glob(self.root_dir+"train/NRG/*.jpg")))) # 78537
            # print(len(sorted(glob.glob(self.root_dir+"val/NRG/*.jpg")))) # 9817
            # print(len(sorted(glob.glob(self.root_dir+"test/NRG/*.jpg")))) # 9818

            # train_data_paths = sorted(glob.glob(self.root_dir+"train/RG/*.jpg"))[:2] + sorted(glob.glob(self.root_dir+"train/NRG/*.jpg"))[:2] # from train path
            # val_data_paths = sorted(glob.glob(self.root_dir+"val/RG/*.jpg"))[:2] + sorted(glob.glob(self.root_dir+"val/NRG/*.jpg"))[:2] # from validation path
            # test_data_paths = sorted(glob.glob(self.root_dir+"test/RG/*.jpg"))[:2] + sorted(glob.glob(self.root_dir+"test/NRG/*.jpg"))[:2] # from test path

            
            
            # FOR DATA OF DIFFERENT SHAPE, ONLY TAKING SHAPE OF 2000-3000
            # df = pd.read_csv("/home/shirshak/BPEye_Project_2024/zzz_tests/df_v2_H_W_Mean-Intensity_labelsv222.csv")
            # df['Height Range'] = pd.cut(df['height'], bins=[df['height'].min()-1, 1000, 2000, 3000, 4000, df['height'].max()+1], labels=[f"{df['height'].min()}-1000","1000-2000", "2000-3000", f"3000-4000", f"4000-{df['height'].max()}"])
            
            # image_locations = df[df['Height Range'] == "2000-3000"].apply(lambda row: os.path.join(self.root_dir, row['train_val_test'], row['label'], row['Image Name']) ,axis=1)
            
            # RG_images = image_locations[image_locations.apply(lambda x: '/RG' in x)].tolist()
            # NRG_images = image_locations[image_locations.apply(lambda x: '/NRG' in x)].tolist()

            # train_data_paths = sorted(RG_images)[:50] + sorted(NRG_images)[:50] # from train path
            # val_data_paths = sorted(RG_images)[50:55] + sorted(NRG_images)[50:55] # from validation path
            # test_data_paths = sorted(RG_images)[30:40] + sorted(NRG_images)[40:50] # from test path

            # print(train_data_paths)
            # print(val_data_paths)
            # print(test_data_paths)

            # self.data_train = BP_Eye_Dataset(train_data_paths, classes=self.classes, my_transforms=my_transforms)
            # self.data_val = BP_Eye_Dataset(val_data_paths, classes=self.classes, my_transforms=my_transforms)
            # self.data_test = BP_Eye_Dataset(test_data_paths, classes=self.classes, my_transforms=my_transforms)

            train_data_paths = os.path.join(self.root_dir+"train/") # from train path
            val_data_paths = os.path.join(self.root_dir+"validation/") # from validation path
            test_data_paths = os.path.join(self.root_dir+"test/") # from test path

            self.data_train = torchvision.datasets.ImageFolder(train_data_paths, transform=my_transforms)
            self.data_val = torchvision.datasets.ImageFolder(val_data_paths, transform=my_transforms)
            self.data_test = torchvision.datasets.ImageFolder(test_data_paths, transform=my_transforms)

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
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,)

if __name__ == "__main__":
    _ = GlaucomaEyepacsModule()