# coding: utf-8

import numpy as np


class SpikingNetwork:
    TAU_MP = 20  # / ms
    T_REF = 1  # / ms
    ALPHA = 2
    ETA_W = 0.002
    ETA_TH = ETA_W * 0.1

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
        #self.weights.append(np.random.uniform(0, root_3_per_m, (n, prev_n)))
        self.thresholds.append(np.ones((n, 1)) * SpikingNetwork.ALPHA * root_3_per_m)

    def forward(self, x):
        self.spikes = [[] for i in range(len(self.v_mps) + 1)]
        for t, xt in enumerate(x):
            input_spike = xt
            for i, (v_mp, spike, weight, threshold) in enumerate(zip(self.v_mps, self.spikes, self.weights, self.thresholds)):
                if not np.any(input_spike):
                    break
                spike.append((t, input_spike))

                t_p = t
                if len(spike) == 1:
                    t_p_1 = 0
                else:
                    t_p_1 = spike[len(spike) - 2][0]

                selected_input_indices = [i for i, value in enumerate(spike[len(spike) - 2][1]) if value]
                selected_weight = weight[:, selected_input_indices]
                sum_weight = selected_weight.sum(axis=1)[:, np.newaxis]

                t_out = np.zeros(v_mp.shape, dtype=int)
                for candidate_t_out, value in self.spikes[i + 1]:
                    t_out[np.where(value & (t_out != 0))] = candidate_t_out
                    if np.all(t_out > 0):
                        break
                w_dyn = ((t_out - t_p) / SpikingNetwork.T_REF) ** 2
                w_dyn[w_dyn > 1] = 1

                v_mp = v_mp * np.exp((t_p_1 - t_p) / SpikingNetwork.TAU_MP) + sum_weight * w_dyn

                input_spike = np.zeros(v_mp.shape, dtype=bool)
                input_spike[v_mp > threshold] = True
                v_mp[v_mp > threshold] -= threshold[v_mp > threshold]

                self.v_mps[i] = v_mp
                self.spikes[i] = spike

            if not np.any(input_spike):
                continue
            self.spikes[len(self.spikes) - 1].append((t, input_spike))
        self.t = t

        for i, _ in enumerate(self.v_mps):
            self.v_mps[i] = np.zeros(self.v_mps[i].shape)

    def backward(self, y):
        for t in range(self.t):
            if t < self.t - 1:
                continue

            '''
            for i, spike in enumerate(self.spikes[1:]):
                if len(spike) == 0:
                    #print('prev weight: {}'.format(self.weights[i]))
                    self.weights[i] += np.abs(np.min(self.weights[i]))
                    #print('post weight: {}'.format(self.weights[i]))
                    return
            '''

            x_ks = [np.array([[np.exp((t_p - t) / SpikingNetwork.TAU_MP) if fire else 0 for fire in value] for t_p, value in spike]).sum(axis=0)[:, np.newaxis]
                    if len(spike) != 0 else np.ones((num_neuron, 1)) * 10 * (1 if i < len(self.weights) - 1 else 1) for i, (spike, num_neuron) in enumerate(zip(self.spikes[:-1], self.num_neurons[:-1]))]
            #x_ks = [sum([np.exp((t_p - t) / SpikingNetwork.TAU_MP) for t_p, _ in spike if t_p <= t]) * np.ones((num_neuron, 1))
            #        for spike, num_neuron in zip(self.spikes[:-1], self.num_neurons[:-1])]
            a_is = [np.array([[np.exp((t_p - t) / SpikingNetwork.TAU_MP) if fire else 0 for fire in value] for t_p, value in spike]).sum(axis=0)[:, np.newaxis]
                    if len(spike) != 0 else np.ones((num_neuron, 1)) * 10 * (1 if i < len(self.weights) - 1 else 1) for i, (spike, num_neuron) in enumerate(zip(self.spikes[1:], self.num_neurons[1:]))]
            #a_is = [sum([np.exp((t_p - t) / SpikingNetwork.TAU_MP) for t_p, _ in spike if t_p <= t]) * np.ones((num_neuron, 1))
            #        for spike, num_neuron in zip(self.spikes[1:], self.num_neurons[1:])]

            sharp_spikes = self._calculate_sharp_spikes()
            if sharp_spikes.max() < 0.0000001:
                #delta = np.ones((self.num_neurons[len(self.num_neurons) - 1], 1))
                #delta = y
                o_is = np.zeros(y.shape)
                no_spike_in_output = True
            else:
                o_is = sharp_spikes / sharp_spikes.max()
                no_spike_in_output = False
            delta = o_is - y
            #delta = y - o_is
            #delta = y - self.infer()
            '''
            if not np.all(delta == delta):
                print('reset weights...')
                self._reset_weights()
                return
            '''
            #print('first delta: {}'.format(delta))

            def get_m_ls(spike):
                spike_t = list(filter(lambda x: x[0] == t, spike))
                return spike_t[0][1].sum() if len(spike_t) != 0 else 0.0001
            m_ls = [get_m_ls(spike) for spike in self.spikes[:-1]]

            for i, weight, threshold, x_k, a_i, m_l in zip(
                    reversed(range(len(m_ls))),
                    reversed(self.weights),
                    reversed(self.thresholds),
                    reversed(x_ks), reversed(a_is),
                    reversed(m_ls)):
                N_l = weight.shape[0]
                M_l = weight.shape[1]

                if no_spike_in_output:
                    if np.sum(delta @ x_k.T) > 0:
                        delta *= -1
                delta_weight = -SpikingNetwork.ETA_W * np.sqrt(N_l / m_l) * delta @ x_k.T
                delta_threshold = -SpikingNetwork.ETA_TH * np.sqrt(N_l / (m_l * M_l)) * delta * a_i
                #if no_spike_in_output:
                #    print('delta_weight: {}'.format(delta_weight))

                if i - 1 >= 0:
                    delta = (1 / self.thresholds[i - 1]) / np.sqrt((1 / m_ls[i - 1]) * np.sum((1 / self.thresholds[i - 1]) ** 2)) * np.sqrt(
                        M_l / m_l) * weight.T @ delta

                weight += delta_weight - 0.0001 * weight
                threshold += delta_threshold

                '''
                if not np.any(weight > 0):
                    print('delta_weight: {}'.format(delta_weight))
                    root_3_per_m = np.sqrt(3 / self.num_neurons[i])
                    weight = np.random.uniform(0, root_3_per_m, weight.shape)
                    self.weights[i] = weight
                    print('layer {}, weight is reset'.format(i))
                '''
                #print('delta_weight: {}'.format(delta_weight))

    def infer(self):
        sharp_spikes = self._calculate_sharp_spikes()
        max_sharp_spike = np.max(sharp_spikes)
        return np.exp(sharp_spikes - max_sharp_spike) / np.sum(np.exp(sharp_spikes - max_sharp_spike))

    def _calculate_sharp_spikes(self):
        return np.array([spike[1] for spike in self.spikes[len(self.spikes) - 1]]).sum(axis=0)

    def _reset_weights(self):
        for i, num_neuron in enumerate(self.num_neurons):
            if i == len(self.weights):
                break

            root_3_per_m = np.sqrt(3 / num_neuron)
            #self.weights[i] = np.random.uniform(-root_3_per_m, root_3_per_m, self.weights[i].shape)
            self.weights[i] = np.random.uniform(0, root_3_per_m, self.weights[i].shape)

