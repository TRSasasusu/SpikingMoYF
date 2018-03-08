# coding: utf-8

import numpy as np
from network import STDPNetwork


def main():
    network = STDPNetwork()
    network.add(1)
    network.add(3)
    network.add(2)
    for i in range(3000):
        if i % 1500 == 0:
            print('{}: {}'.format(i, network.voltages))
        network.execute(np.sin(np.arange(0, 1, 0.1))[:, np.newaxis])
    print('results: {}'.format(network.voltages))
    print('weights: {}'.format(network.weights))


if __name__ == '__main__':
    main()
