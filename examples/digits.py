# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from PIL import Image
from network import SpikingNetwork


def main():
    network = SpikingNetwork()
    network.add(64)
    network.add(128)
    network.add(32)
    network.add(10)

    train_data = []
    test_data = []

    imgs_path = 'imgs/digits/'
    for img_name in os.listdir(imgs_path):
        img = Image.open(os.path.join(imgs_path, img_name), 'r')
        number_str = img_name[0]
        number = np.zeros((10, 1))
        number[int(number_str)] = 1

        appended_data = {
            'x': [np.array([img.getpixel((x, y))[0] == 255 for y in range(64)])[:, np.newaxis] for x in range(64)],
            'y': number
        }
        if int(number_str) == len(test_data):
            test_data.append(appended_data)
        else:
            train_data.append(appended_data)

    for i in range(1000):
        if i % 10 == 0:
            for data in test_data:
                network.forward(data['x'])
                print('{}: {}'.format(np.where(data['y'])[0], network.infer()))
        random.shuffle(train_data)
        for data in train_data:
            network.forward(data['x'])
            network.backward(data['y'])


if __name__ == '__main__':
    main()
