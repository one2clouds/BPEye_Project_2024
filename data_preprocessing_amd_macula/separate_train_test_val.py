import pandas as pd
import os 
from glob import glob 
import random
import shutil 
from tqdm import tqdm 

def separate_val_and_test(train_val_test):
    if train_val_test == 'Validation':
        df = pd.read_csv('/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Validation/validation_classification_GT.txt', sep=' ', header=None, names=["Filename", "Label"])
    elif train_val_test == 'Test':
        df = pd.read_csv('/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Test/test_classification_GT.txt', sep='  ', header=None, names=["Filename", "Label"])

    test_non_amd, val_non_amd, test_amd, val_amd = [], [], [], []
    
    for img_path in glob(f'/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/{train_val_test}/image/*.jpg'):    
        # print(img_path)
        # print(str(os.path.split(img_path)[1]))
        if df.loc[df['Filename'] == str(os.path.split(img_path)[1]), 'Label'].item() == 0:
            if train_val_test == 'Test':
                test_non_amd.append(img_path)
            elif train_val_test == 'Validation':
                val_non_amd.append(img_path)
            else:
                print("NO TEST OR VALIDATION DEFINED")
                continue
        else:
            if train_val_test == 'Test':
                test_amd.append(img_path)
            elif train_val_test == 'Validation':
                val_amd.append(img_path)
            else:
                print("NO TEST OR VALIDATION DEFINED ")
                continue

    if train_val_test == 'Validation':
        return val_non_amd, val_amd
    if train_val_test == 'Test':
        return test_non_amd, test_amd



def copy_files_using_train_val_test_splits(val_or_test_folder_out, train_folder_out, val_or_test_non_or_amd, val_or_test_counts, amd_or_non_amd):
    for idx, img_path in enumerate(tqdm(val_or_test_non_or_amd)):
        if idx <= val_or_test_counts:
            dest_path = os.path.join(val_or_test_folder_out, amd_or_non_amd, os.path.split(img_path)[1])
        else: 
            dest_path = os.path.join(train_folder_out, amd_or_non_amd, os.path.split(img_path)[1])
        # print(img_path)
        # print(dest_path)
        shutil.copy(img_path, dest_path)

def copy_files_for_train_split_only(train_folder_out, train_not_amd_or_amd, amd_or_non_amd):
    for img_path in tqdm(train_not_amd_or_amd):
        dest_path = os.path.join(train_folder_out, amd_or_non_amd, os.path.split(img_path)[1])
        # print(img_path)
        # print(dest_path)
        shutil.copy(img_path, dest_path)
    
        

if __name__ == "__main__":
    val_df = pd.read_csv('/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Validation/validation_classification_GT.txt', sep=' ', header=None, names=["Filename", "Label"])
    test_df = pd.read_csv('/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Test/test_classification_GT.txt', sep='  ', header=None, names=["Filename", "Label"])

    train_val_test = 'Validation'
    val_non_amd, val_amd = separate_val_and_test(train_val_test)

    train_val_test = 'Test'
    test_non_amd, test_amd = separate_val_and_test(train_val_test)

    # train + test + val = total, test = 0.1*total and val = 0.1 * total. 
    val_non_amd_count = 94
    val_amd_count = 27
    test_non_amd_count = 94
    test_amd_count = 27

    random.shuffle(val_non_amd)
    random.shuffle(val_amd)
    random.shuffle(test_non_amd)
    random.shuffle(test_amd)

    output_folder = "/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/train_test_val_separated_ADAM/"
    train_folder_out = os.path.join(output_folder, "train")
    val_folder_out = os.path.join(output_folder, "val")
    test_folder_out = os.path.join(output_folder, "test")

    for folder in [train_folder_out, val_folder_out, test_folder_out]:
        os.makedirs(os.path.join(folder, "AMD"), exist_ok=True)
        os.makedirs(os.path.join(folder, "Non-AMD"), exist_ok=True)


    copy_files_using_train_val_test_splits(val_folder_out, train_folder_out, val_non_amd, val_non_amd_count, "Non-AMD")
    copy_files_using_train_val_test_splits(val_folder_out, train_folder_out, val_amd, val_amd_count, "AMD")
    copy_files_using_train_val_test_splits(test_folder_out, train_folder_out, test_non_amd, test_non_amd_count, "Non-AMD")
    copy_files_using_train_val_test_splits(test_folder_out, train_folder_out, test_amd, test_amd_count, "AMD")


    train_non_amd = glob("/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Train/Training-image-400/Non-AMD/*.jpg")
    train_amd = glob("/mnt/Enterprise2/shirshak/ADAM_AgeRelatedMacularDegeneration/ADAM/Train/Training-image-400/AMD/*.jpg")

    copy_files_for_train_split_only(train_folder_out, train_non_amd, "Non-AMD")
    copy_files_for_train_split_only(train_folder_out, train_amd, "AMD")










