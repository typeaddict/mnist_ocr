import argparse
import os

import keras
import tensorflow as tf
import idx2numpy
import numpy as np
from keras import backend as K

from get_train_data import get_data
from model import emnist_model
from utils import labels



parser = argparse.ArgumentParser(description='OCR Train Module')
parser.add_argument('--dataset_path', default='dataset/gzip/', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=1, type=int, help='amount of training epochs')
parser.add_argument('--output_path', default='weights/', type=str, help='output path to store trained model')
args = parser.parse_args()

def main():
    model = emnist_model(labels)
    
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3,
                                                                verbose=1, factor=0.5, min_lr=0.00001)

    keras.backend.get_session().run(tf.global_variables_initializer())

    X_train, X_test, x_train_cat, y_test_cat = get_data(args.dataset_path, labels)
    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
                                                     batch_size=args.batch_size, epochs=args.epochs)

    model.save(os.path.join(args.output_path, 'emnist_letters.h5'))
        
if __name__ == '__main__':
    main()
