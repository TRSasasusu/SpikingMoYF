# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from sklearn import datasets
from network import SpikingNetwork


MAX_DIGIT = 3


def convert_image(image):
    return np.array([[x > 0 for j, x in enumerate(y) if j % 2 == 0] for i, y in enumerate(image) if i % 2 == 0]).reshape((196, 1))


def make_number(number):
    zeros = np.zeros((MAX_DIGIT, 1))
    zeros[number] = 1
    return zeros


def main():
    network = SpikingNetwork()
    network.add(196)
    #network.add(24)
    #network.add(12)
    network.add(50, -0.5)
    network.add(6, -0.9)
    network.add(MAX_DIGIT, -0.3)

    train_data = []
    test_data = []

    print('MNIST Loading...', end='')
    mnist = datasets.fetch_mldata('MNIST original')
    print('OK')

    for number in range(MAX_DIGIT):
        both_data = [{
            'x': convert_image(data.reshape((28, 28))),
            'y': make_number(number)
        } for data in mnist['data'][mnist['target'] == number]]
        train_data.extend(both_data[:30])
        test_data.extend(both_data[30:35])

    #test_data = [train_data[0], train_data[1]]
    #import bpdb; bpdb.set_trace()

    for i in range(10000):
        if i % 10 == 0:
            print('In {}'.format(i))
            network.forward(np.concatenate([data['x'] for data in test_data], axis=1), 50)
            print('answer: {}'.format(np.concatenate([data['y'] for data in test_data], axis=1)))
            print('infer:  {}'.format(network.infer()))
        random.shuffle(train_data)
        network.forward(np.concatenate([data['x'] for data in train_data], axis=1), 50)
        network.backward(np.concatenate([data['y'] for data in train_data], axis=1))

        '''
        end = True
        for data in test_data:
            network.forward(data['x'], 50)
            infer = network.infer()
            if isinstance(infer, float):
                end = False
                break
            if infer[np.where(data['y'])[0]] < 0.8:
                end = False
                break
        if end:
            print('great!')
            for data in train_data:
                network.forward(data['x'], 50)
                print('{}: {}'.format(np.where(data['y'])[0], network.infer()))
            return
        '''


if __name__ == '__main__':
    main()
