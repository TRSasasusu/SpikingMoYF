# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


class SpikingNetwork:
    TAU_MP = 20  # / ms
    T_REF = 1 # / ms
    ALPHA = 5

    def __init__(self):
        self.weights = []
        self.thresholds = []
        self.num_neurons = []
        self.v_mps = []

    def add(self, n):
        self.num_neurons.append(n)
        if len(self.num_neurons) == 1:
            return

        self.v_mps.append(np.zeros((n, 1)))

        prev_n = self.num_neurons[len(self.num_neurons) - 2]
        root_3_per_m = np.sqrt(3 / prev_n)

        self.weights.append(np.random.uniform(-root_3_per_m, root_3_per_m, (n, prev_n)))
        self.thresholds.append(np.ones((n, 1)) * SpikingNetwork.ALPHA * root_3_per_m)

    def forward(self, x, y):
        spikes = [[] for v_mp in self.v_mps]
        for t, xt in enumerate(x):
            input_spike = xt
            for i, (v_mp, spike, weight, threshold) in enumerate(zip(self.v_mps, spikes, self.weights, self.thresholds)):
                if not np.any(input_spike):
                    break
                spike.append((t, input_spike))

                t_p = t
                if len(spike) == 1:
                    t_p_1 = 0
                else:
                    t_p_1 = spike[len(spike) - 2][0]

                selected_input_indices = [i for i, value in enumerate(spike[len(spike) - 2][1]) if value]
                selected_weight = weight[selected_input_indices]
                mean_weight = selected_weight.mean(axis=0)[np.newaxis, :].T

                t_out = np.zeros(v_mp.shape, dtype=int)
                for candidate_t_out, value in spike[i + 1]:
                    t_out[np.where(value & (t_out != 0))] = candidate_t_out
                    if np.all(t_out > 0):
                        break
                w_dyn = ((t_out - t_p) / SpikingNetwork.T_REF) ** 2
                w_dyn[w_dyn > 1] = 1

                v_mp = v_mp * np.exp((t_p_1 - t_p) / SpikingNetwork.TAU_MP) + mean_weight * w_dyn

                input_spike = np.zeros(v_mp.shape, dtype=bool)
                input_spike[v_mp > threshold] = True
                v_mp[v_mp > threshold] -= threshold

            if not np.any(input_spike):
                break
            spikes[len(spikes) - 1].append((t, input_spike))

    def backward(self):
