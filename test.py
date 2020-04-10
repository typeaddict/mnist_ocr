import os
import argparse

import keras
import numpy as np

from utils import emnist_predict_img, img_to_str

parser = argparse.ArgumentParser(description='OCR Test Module')
parser.add_argument('--image_path', default='test_images/img.jpg', type=str, help='image path')
parser.add_argument('--model_path', default='weights/emnist_letters.h5', type=str, help='pretrained model')
args = parser.parse_args()


if __name__ == '__main__':
    
    model = keras.models.load_model(args.model_path)
    s_out = img_to_str(model, args.image_path)
    print(s_out)

