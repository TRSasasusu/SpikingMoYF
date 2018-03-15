# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from PIL import Image
from network import SpikingNetwork


MAX_DIGIT = 2


def main():
    network = SpikingNetwork()
    network.add(8)
    #network.add(24)
    #network.add(12)
    network.add(6)
    network.add(4)
    network.add(MAX_DIGIT)

    train_data = []
    test_data = []

    imgs_path = 'imgs/digits/'
    for img_name in os.listdir(imgs_path):#['0_0.png', '1_0.png']:
        img = Image.open(os.path.join(imgs_path, img_name), 'r')
        number_str = img_name[0]
        if number_str == str(MAX_DIGIT):
            break
        number = np.zeros((MAX_DIGIT, 1))
        number[int(number_str)] = 1

        appended_data = {
            'x': [np.array([img.getpixel((x, y))[0] != 255 for x in range(64) if x % 8 == 0])[:, np.newaxis] for y in range(64) if y % 8 == 0],
            'y': number
        }
        if int(number_str) == len(test_data):
            test_data.append(appended_data)
        else:
            train_data.append(appended_data)

    #test_data = [train_data[0], train_data[1]]
    #import bpdb; bpdb.set_trace()

    for i in range(100000):
        if i % 100 == 0:
            print('In {}'.format(i))
            for data in test_data:
                network.forward(data['x'])
                print('{}: {}'.format(np.where(data['y'])[0], network.infer()))
        random.shuffle(train_data)
        for data in train_data:
            network.forward(data['x'])
            network.backward(data['y'])


if __name__ == '__main__':
    main()
