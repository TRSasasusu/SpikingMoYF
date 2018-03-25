# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from sklearn import datasets
from network import SpikingNetwork


MAX_DIGIT = 3
MINI_BATCH_SIZE = 20


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
        train_data.extend(both_data[:100])
        test_data.extend(both_data[100:105])
        #train_data.extend(both_data[:1])
        #test_data.extend(both_data[:1])

    if '--load' in sys.argv[1:]:
        print('Loading...', end='')
        network.load()
        print('OK')

    for i in range(10000):
        network.forward(np.concatenate([data['x'] for data in test_data], axis=1), 50)
        answer = np.concatenate([data['y'] for data in test_data], axis=1)
        infer = network.infer(display_no_spike=i % 10 == 0)
        complete = False
        if np.all(np.absolute(infer - answer) < 0.1):
            complete = True
        if i % 10 == 0 or complete:
            print('In {}'.format(i))
            print('answer:\n{}'.format(answer))
            print('infer:\n{}'.format(infer))
        if complete:
            print('Complete!')
            if '--no-save' not in sys.argv[1:]:
                print('Saving...', end='')
                network.save()
                print('OK')
            return

        random.shuffle(train_data)
        for j in range(0, len(train_data), MINI_BATCH_SIZE):
            network.forward(np.concatenate([data['x'] for data in train_data[j:j + MINI_BATCH_SIZE]], axis=1), 50)
            network.backward(np.concatenate([data['y'] for data in train_data[j:j + MINI_BATCH_SIZE]], axis=1))


if __name__ == '__main__':
    main()
