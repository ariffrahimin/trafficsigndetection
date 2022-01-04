from sklearn.model_selection import train_test_split
from my_utils import split_data, order_test_set

if __name__=="__main__":
    if False:
        path_to_data = "E:\\Projects\\Python\\trafficsigndetection\\traffic_sign_data\\Train"
        path_to_save_train = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\train"
        path_to_save_val = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\val"
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    path_to_images = "E:\\Projects\\Python\\trafficsigndetection\\traffic_sign_data\\Test"
    path_to_csv = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\Test.csv"
    order_test_set(path_to_images, path_to_csv)