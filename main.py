# coding: utf-8

import numpy as np
from network import SpikingNetwork


def main():
    network = SpikingNetwork()
    network.add(5)
    network.add(8)
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

    for i in range(100):
        network.forward(rects)
        network.backward(np.array([[1], [0]]))
        network.forward(tris)
        network.backward(np.array([[0], [1]]))

    network.forward(rects)
    print(network.infer())
    network.forward(tris)
    print(network.infer())


if __name__ == '__main__':
    main()
