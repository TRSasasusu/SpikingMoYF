# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


class STDPNetwork:
    VOLTAGE_THRESHOLD = 1.5
    MEMBRANE_VOLTAGE = 1
    UPDATE_COEFFICIENT = 0.01

    def __init__(self):
        self.weights = []
        self.biases = []
        self.voltages = []

    def add(self, n):
        self.voltages.append(np.random.sample((n, 1)))
        if len(self.voltages) == 1:
            return

        prev_n = self.voltages[len(self.voltages) - 2].shape[0]

        self.weights.append(np.random.sample((n, prev_n)) * 2 - 1)  # (-1, 1)
        self.biases.append(np.zeros((n, 1)))

    def execute(self, x):
        fire_times = [np.zeros(voltage.shape) for voltage in self.voltages]
        for t, xt in enumerate(x):
            #            print('self.voltages: {}'.format(self.voltages))
            input_voltage = xt[:, np.newaxis]
#            print('input_voltage: {}'.format(input_voltage))
            prev_fire_time = None
            for voltage, weight, bias, fire_time in zip(self.voltages, self.weights, self.biases, fire_times):
                #                print('prev v: {}'.format(voltage))
                voltage += -(voltage - STDPNetwork.MEMBRANE_VOLTAGE) + input_voltage
                voltage[voltage < 0] = 0
#                print('post v: {}'.format(voltage))

                input_voltage = voltage.copy()
                input_voltage[input_voltage < STDPNetwork.VOLTAGE_THRESHOLD] = 0

                fire_time[input_voltage > 0] = t
                voltage[input_voltage > 0] = 0

                input_voltage[input_voltage > 0] = STDPNetwork.VOLTAGE_THRESHOLD
#                print('weight.shape: {}'.format(weight.shape))
#                print('input_voltage.shape: {}'.format(input_voltage.shape))
                input_voltage = weight.dot(input_voltage) + bias

#                print('prev weight: {}'.format(weight))
                if prev_fire_time is not None:
                    prev_fire_worked = np.zeros(prev_fire_time.shape, bool)
                    fire_worked = np.zeros(fire_time.shape, bool)
                    for ft, weight_row, fw in zip(fire_time, weight, fire_worked):
                        if ft[0] == 0:
                            continue
                        for pft, i, pfw in zip(prev_fire_time, range(len(weight_row)), prev_fire_worked):
                            if pft[0] == 0:
                                continue
                            delta_t = pft - ft
                            if delta_t < 0:
                                update = STDPNetwork.UPDATE_COEFFICIENT * np.exp(delta_t)
                            else:
                                update = -STDPNetwork.UPDATE_COEFFICIENT * np.exp(-delta_t)
#                            print('prev weight_elem: {}'.format(weight_row[i]))
                            weight_row[i] += update
#                            print('post weight_elem: {}'.format(weight_row[i]))
                            fw[0] = True
                            pfw[0] = True
#                    print('prev_fire_worked: {}'.format(prev_fire_worked))
                    prev_fire_time[prev_fire_worked] = 0
                prev_fire_time = fire_time
#                print('post weight: {}'.format(weight))

            voltage = self.voltages[len(self.voltages) - 1]
            voltage += -(voltage - STDPNetwork.MEMBRANE_VOLTAGE) + input_voltage
            voltage[voltage > STDPNetwork.VOLTAGE_THRESHOLD] = 0
            voltage[voltage < 0] = 0

            fire_time[fire_worked] = 0
