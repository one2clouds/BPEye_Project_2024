import pandas as pd
import os 
import shutil 
from glob import glob 

def preprocess_data():
    df = pd.read_csv('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/train_labels.csv')
    for img_path in sorted(glob(f'/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/overall_data/*/*/*.jpg')):    
        # print(str(os.path.split(img_path)[1]).split('.jpg')[0])
        # print(df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item())
        if df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item() == 'NRG':
            print(os.path.join('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_overall_data/NRG/', os.path.split(img_path)[1]))
            shutil.copy(img_path, os.path.join('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_overall_data/NRG/', os.path.split(img_path)[1])) 

        elif df.loc[df['challenge_id'] == os.path.split(img_path)[1].split('.jpg')[0], 'class'].item() == 'RG':
            print(os.path.join('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_overall_data/RG/', os.path.split(img_path)[1]))
            shutil.copy(img_path, os.path.join('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_overall_data/RG/', os.path.split(img_path)[1])) 
        

if __name__ == "__main__":
    preprocess_data()