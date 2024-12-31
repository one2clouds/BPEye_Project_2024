import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm 



def split_and_copy(files, source_folder, train_dest, val_dest, test_dest):
    train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    for file in tqdm(train_files):
        shutil.copy(os.path.join(source_folder, file), train_dest)
    for file in val_files:
        shutil.copy(os.path.join(source_folder, file), val_dest)
    for file in test_files:
        shutil.copy(os.path.join(source_folder, file), test_dest)


if __name__== '__main__':
    data_folder = "/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_overall_data/"
    glaucoma_folder = os.path.join(data_folder, "RG")
    non_glaucoma_folder = os.path.join(data_folder, "NRG")

    output_folder = "/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_separated_train_test_val/"
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    test_folder = os.path.join(output_folder, "test")

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(os.path.join(folder, "RG"), exist_ok=True)
        os.makedirs(os.path.join(folder, "NRG"), exist_ok=True)

    glaucoma_files = os.listdir(glaucoma_folder)
    split_and_copy(
        glaucoma_files,
        glaucoma_folder,
        os.path.join(train_folder, "RG"),
        os.path.join(val_folder, "RG"),
        os.path.join(test_folder, "RG"),
    )

    non_glaucoma_files = os.listdir(non_glaucoma_folder)
    split_and_copy(
        non_glaucoma_files,
        non_glaucoma_folder,
        os.path.join(train_folder, "NRG"),
        os.path.join(val_folder, "NRG"),
        os.path.join(test_folder, "NRG"),
    )

    





