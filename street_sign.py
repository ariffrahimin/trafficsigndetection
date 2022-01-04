import os
import glob
from sklearn.model_selection import train_test_split
def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(images_paths,test_size=split_size)

        for x in x_train:

            basename = os.path.basename(x)
            path_to_folder = os.path.join(path_to_save_train, folder)