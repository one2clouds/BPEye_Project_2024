import torchvision
from glob import glob
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    
    images = sorted(glob('/mnt/Enterprise2/shirshak/GLAUCOMA_DATASET_EYEPACS_AIROGS_ZENODO/preprocessed_cropped_separated_train_test_val/*/*/*.jpg'))
    df = pd.DataFrame(columns=["Image Name", "height", "width", "label", "train_val_test", "Mean Intensity Value", "Mean Intensity Value R,G,B"])

    for index, image in enumerate(tqdm(images)):

        img_name = image.split("/")[-1]
        height = torchvision.io.read_image(image).shape[1]
        width = torchvision.io.read_image(image).shape[2]
        train_val_test = image.split("/")[-3]
        label = image.split("/")[-2]
        mean_intensity = torchvision.io.read_image(image).float().mean()
        mean_intensity_per_channel = torchvision.io.read_image(image).float().mean(dim=[1, 2])
        df.loc[index] = {
            "Image Name": img_name,
            "height": height,
            "width": width, 
            "label": label, 
            "train_val_test": train_val_test,
            "Mean Intensity Value": mean_intensity.item(),
            "Mean Intensity Value R,G,B": mean_intensity_per_channel.tolist()
            }
    print(df)
    # df.to_csv("/home/shirshak/BPEye_Project_2024/zzz_tests/df_v3_H_W_Mean-Intensity_labelsv333.csv", index=False)





    