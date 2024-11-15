from ultralytics import YOLO


def Trainer(model_weights : str ,dataset_file : str, batch_size : int, epochs : int, imgsz : int, patience : int):

    model = YOLO(model_weights)
    try:
        if dataset_file is not None:
            model.train(data=dataset_file,batch = batch_size,  epochs=epochs, imgsz=imgsz, patience = patience)

    except Exception as FileNotFoundError:
        print("Dataset file not Found")
        exit()


if __name__ == "__main__":
    # Place data inside data folder in form of Train and Test
    # Train and Test folders contain images and labels folder e.g data/Train/images/1.jpg, 2.jpg.... , data/Train/labels/1.txt, 2.txt
    # Edit Dataset Yaml file --> write path: data, train: data/Train/images, val: data/Test/images
    # In same dataset.yaml file --> classes , 0: clean_panel , 1: dirty_panel

    MODEL_WEIGHTS = "yolo11s.pt"
    BATCH_SIZE = 16
    EPOCHS = 30
    IMAGE_SIZE = 640
    patience_level = 10

    Trainer(MODEL_WEIGHTS, "dataset.yaml" , batch_size=BATCH_SIZE, epochs=EPOCHS , imgsz=IMAGE_SIZE, patience=patience_level)