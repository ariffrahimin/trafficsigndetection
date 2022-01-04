from sklearn.model_selection import train_test_split
from my_utils import split_data

if __name__=="__main__":
    path_to_data = "E:\\Projects\\Python\\trafficsigndetection\\traffic_sign_data\\Train"
    path_to_save_train = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\train"
    path_to_save_val = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\val"
    split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)