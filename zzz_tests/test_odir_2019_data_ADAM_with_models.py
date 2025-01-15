import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import torch 
from tqdm import tqdm 
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pathlib
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd 
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import argparse
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights
import random
import rootutils
import time



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
        image = Image.open(image_filepath)
        
        image = self.my_transforms(image)
        label_final = self.class_to_idx[label]

        # print(label) # RG->1 , NRG->0
        # print(label_final)

        return image, label_final, image_filepath


if __name__ == "__main__":
    
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ['Non-AMD', 'AMD']

    # model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    # checkpoint = torch.load('/home/shirshak/BPEye_Project_2024/logs/train/runs/5) glaucoma_efficientnet_airogs_CE_Loss/checkpoints/epoch_004.ckpt')
    # checkpoint = torch.load('/home/shirshak/BPEye_Project_2024/logs/train/runs/3) glaucoma_efficientnet_airogs_FOCAL/checkpoints/epoch_012.ckpt')


    model = torchvision.models.resnet50(weights=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    checkpoint = torch.load('/home/shirshak/BPEye_Project_2024/logs/train/runs/6) amd_macula_preprocessed_resnet_focal/checkpoints/epoch_011.ckpt')
    
    # print(checkpoint.keys())
    # model.load_state_dict(checkpoint['state_dict'])

    state_dict = {k.replace('net.model.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    my_transforms = transforms.Compose([
        # transforms.ToPILImage(), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
        transforms.Resize(size=(512,512),antialias=True),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    test_data_paths = sorted(glob('/mnt/Enterprise2/shirshak/Ocular_Disease_Intelligent_Recognition_ODIR-2019/preprocessed_and_cropped_ODIR_2019_AMD_DATA/*/*/*'))
    # doing shuffling in the val data because, at a batch there can be all negative case, & even if model is ideal, the metrics like precision, recall is 0%
    # shuffling it now, and will not shuffle it later on during data loading......
    random.shuffle(test_data_paths)
    data_test = BP_Eye_Dataset(test_data_paths, classes=classes, my_transforms=my_transforms)

    test_dataloader = DataLoader(dataset=data_test,batch_size=32,num_workers=1, shuffle=False)

    wrong_cases, right_cases = [], []

    labels_list, predicted_list = [], []

    with torch.no_grad():
        for inputs, labels, image_path in tqdm(test_dataloader):
            # start_time= time.time()
            # inputs, labels = inputs.to(device), labels.to(device)
            # outputs = nn.Sigmoid()(model(inputs)) 
            # stop_time=time.time()
            # duration = stop_time - start_time
            # print(f"Time duration needed for output {outputs.shape} is : {duration}")
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = nn.Sigmoid()(model(inputs)) 
            _, predicted_test = torch.max(outputs, 1)

            labels_list.extend(labels)
            predicted_list.extend(predicted_test)

            wrong_cases += [(x_w, y_w, yp_w, img_path) for x_w, y_w, yp_w, img_path in zip(inputs, labels, predicted_test, image_path) if y_w != yp_w]
            right_cases += [(x_w, y_w, yp_w, img_path) for x_w, y_w, yp_w, img_path in zip(inputs, labels, predicted_test, image_path) if y_w == yp_w]


    accuracy = accuracy_score(labels_list.cpu().numpy(), predicted_list.cpu().numpy())
    precision = precision_score(labels_list.cpu().numpy(), predicted_list.cpu().numpy(), average='weighted', zero_division=0.0)
    recall = recall_score(labels_list.cpu().numpy(), predicted_list.cpu().numpy(), average='weighted', zero_division=0.0)
    f1 = f1_score(labels_list.cpu().numpy(), predicted_list.cpu().numpy(), average='weighted', zero_division=0.0)
    # roc_auc_score_ = roc_auc_score(labels_list.cpu().numpy(), predicted_list.cpu().numpy(), average='weighted')

    confusion_matrix_chart = confusion_matrix(labels_list.cpu().numpy(), predicted_list.cpu().numpy())
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_chart, display_labels = ['Non-AMD', 'AMD'])
    
    cm_display.plot()
    plt.savefig('/home/shirshak/BPEye_Project_2024/zzz_tests/CM_odir-2019.png', dpi=500)
    plt.close()


    labels = ['Non-AMD', 'AMD']

    print(len(wrong_cases))
    print(len(right_cases))

    for count, wrong_case in enumerate(wrong_cases[:25]):
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,5))

        ax[0].imshow(Image.open(wrong_case[3]))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[0].text(x=0.5, y=-0.1, s=f"Real: {labels[wrong_case[1]]}  \n Predicted: {labels[wrong_case[2]]}", ha='center', va='center', transform=ax[0].transAxes, fontsize=10)
        
        ax[1].imshow(torchvision.transforms.ToPILImage()(wrong_case[0]))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

        plt.savefig(f"/home/shirshak/BPEye_Project_2024/zzz_tests/wrong_cases_odir/wrong{count}.jpg")
        plt.close()

    for count, right_case in enumerate(right_cases[:25]):
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,5))

        ax[0].imshow(Image.open(right_case[3]))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[0].text(x=0.5, y=-0.1, s=f"Real: {labels[right_case[1]]}  \n Predicted: {labels[right_case[2]]}", ha='center', va='center', transform=ax[0].transAxes, fontsize=10)
        
        ax[1].imshow(torchvision.transforms.ToPILImage()(right_case[0]))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"/home/shirshak/BPEye_Project_2024/zzz_tests/right_cases_odir/right{count}.jpg")
        plt.close()


    print("Hello, model loaded")
