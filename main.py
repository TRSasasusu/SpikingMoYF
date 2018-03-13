# coding: utf-8

import numpy as np
from network import SpikingNetwork


def main():
    network = SpikingNetwork()
    network.add(5)
    network.add(8)
    network.add(6)
    network.add(2)

    rects = []
    for _ in range(10):
        for _2 in range(5):
            rects.append(np.array([0, 0, 0, 0, 1], dtype=bool)[:, np.newaxis])
        rects.append(np.array([1, 1, 1, 1, 1], dtype=bool)[:, np.newaxis])
        for _2 in range(5):
            rects.append(np.array([1, 0, 0, 0, 0], dtype=bool)[:, np.newaxis])
        rects.append(np.array([1, 1, 1, 1, 1], dtype=bool)[:, np.newaxis])

    tris = []
    for _ in range(10):
        tris.append(np.array([1, 1, 1, 1, 1], dtype=bool)[:, np.newaxis])
        tris.append(np.array([0, 1, 0, 0, 0], dtype=bool)[:, np.newaxis])
        tris.append(np.array([0, 0, 1, 0, 0], dtype=bool)[:, np.newaxis])
        tris.append(np.array([0, 0, 0, 1, 0], dtype=bool)[:, np.newaxis])
        tris.append(np.array([0, 0, 0, 0, 1], dtype=bool)[:, np.newaxis])

    for i in range(1000):
        if i % 10 == 0:
            network.forward(rects)
            print('In {}, rect: {}'.format(i, network.infer()))
            network.forward(tris)
            print('In {}, tri: {}'.format(i, network.infer()))
            if isinstance(network.infer(), float):
                print('weights: {}'.format(network.weights))
        network.forward(rects)
        network.backward(np.array([[1], [0]]))
        network.forward(tris)
        network.backward(np.array([[0], [1]]))

    network.forward(rects)
    print(network.infer())
    network.forward(tris)
    print(network.infer())
    print('last weights: {}'.format(network.weights))
    print('last thresholds: {}'.format(network.thresholds))


def submain():
    network = SpikingNetwork()
    network.add(2)
    network.add(2)

    print('initial weights: {}'.format(network.weights))
    ups = [np.array([[1], [0]], dtype=bool) for _ in range(20)]
    downs = [np.array([[0], [1]], dtype=bool) for _ in range(20)]
    for i in range(100):
        if i % 10 == 0:
            network.forward(ups)
            print('In {}, up: {}'.format(i, network.infer()))
            network.forward(downs)
            print('In {}, down: {}'.format(i, network.infer()))
        network.forward(ups)
        network.backward(np.array([[1], [0]]))
        network.forward(downs)
        network.backward(np.array([[0], [1]]))
    print('end weights: {}'.format(network.weights))


if __name__ == '__main__':
    main()
