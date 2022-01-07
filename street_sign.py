from sklearn.model_selection import train_test_split
from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__=="__main__":


    path_to_train = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\train"
    path_to_val = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\training_data\\val"
    path_to_test = "E:\\Projects\Python\\trafficsigndetection\\traffic_sign_data\\Test"
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    path_to_save_model = './Models'

    ckpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    early_stop = EarlyStopping(monitor="vall_accuracy", patience=10)

    model = streetsigns_model(nbr_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop]
        )