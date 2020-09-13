from tensorflow.keras import models, preprocessing
from mobilenetV3 import MobileNetV3Small
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


IMAGE_SIZE = (224, 224)
ANNOTATIONS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def preprocess_img(img):
    img = img / 127.5 - 1.
    return img


def test_data(df_data):
    datagen = preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_img)
    test_images = datagen.flow_from_dataframe(df_data, x_col='filename', class_mode=None, target_size=IMAGE_SIZE, shuffle=False)

    model = MobileNetV3Small(10).biuld_model()
    model.load_weights('src/model/')
    predictions = model.predict(test_images, verbose=1)

    predictions = np.apply_along_axis(np.argmax, 1, predictions)

    df_data['class'] = pd.DataFrame(predictions, columns=['class'])['class'].apply(lambda x: ANNOTATIONS.get(int(x)))

    return df_data


def get_data_from_folder(path):
    info_dir = os.walk(path)
    paths = pd.DataFrame([], columns=['filename', 'class'])

    for (dir1, dir2, filenames) in info_dir:
        if filenames != []:
            df = pd.DataFrame(filenames, columns=['filename'])
            df = df[df['filename'].str.contains('.jpg$|.png$')]
            df['filename'] = df['filename'].apply(lambda x: os.path.join(path, os.path.join(dir1, x)))
            paths = paths.append(df, ignore_index=True)
    return paths


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess large image for object detection')
    parser.add_argument('folder', type=dir_path)
    args = parser.parse_args()

    df_data = get_data_from_folder(args.folder)

    results = test_data(df_data)

    results.to_csv('results.csv', index=False)
