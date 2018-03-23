# coding: utf-8

import numpy as np


class SpikingNetwork:
    TAU_MP = 20  # / ms
    T_REF = 1  # / ms
    ALPHA = 2
    ETA_W = 0.002
    ETA_TH = ETA_W * 0.1
    SIGMA = 0.5
    #BETA = 0.0000000000001
    #LAMBDA = 0.000000000001

    def __init__(self):
        self.weights = []
        self.thresholds = []
        self.num_neurons = []
        self.v_mps = []
        self.kappas = []

    def add(self, n, kappa=None):
        self.num_neurons.append(n)
        if len(self.num_neurons) == 1:
            if kappa is not None:
                print('This kappa is ignored.')
            return

        self.v_mps.append(np.zeros((n, 1)))

        if kappa is None:
            kappa = 0
        self.kappas.append(kappa)

        prev_n = self.num_neurons[len(self.num_neurons) - 2]
        root_3_per_m = np.sqrt(3 / prev_n)

        self.weights.append(np.random.uniform(-root_3_per_m, root_3_per_m, (n, prev_n)))
        self.thresholds.append(np.ones((n, 1)) * SpikingNetwork.ALPHA * root_3_per_m)

    # time / ms
    def forward(self, x, time):
        self.spikes = [[[] for _j in range(x.shape[1])] for _i in self.num_neurons]
        #self.spikes = [[] for i in range(len(self.v_mps) + 1)]
        self.x_ks = [np.zeros((num_neuron, 1)) for num_neuron in self.num_neurons[:-1]]
        self.a_is = [np.zeros((num_neuron, 1)) for num_neuron in self.num_neurons[1:]]

        for t in range(time):
            input_spike = x
            for i, (v_mp, spike, weight, threshold, x_k, a_i, kappa) in enumerate(zip(
                    self.v_mps,
                    self.spikes,
                    self.weights,
                    self.thresholds,
                    self.x_ks,
                    self.a_is,
                    self.kappas
            )):
                #if not np.any(input_spike):
                #    break
                for unit_spike in spike:
                    unit_spike.append((t, input_spike[:, 0][:, np.newaxis]))

                x_k = np.concatenate([SpikingNetwork._calc_x_k(unit_spike, t) for unit_spike in spike], axis=1)
                if i > 0:
                    self.a_is[i - 1] = x_k

                tmp_x_k = x_k.copy()
                tmp_x_k[~input_spike] = 0
                lateral_inhibition = SpikingNetwork.SIGMA * threshold * kappa * a_i
                lateral_inhibition = lateral_inhibition.sum(axis=0) * np.ones(a_i.shape) - lateral_inhibition
                v_mp = weight @ tmp_x_k - threshold * a_i + lateral_inhibition

                input_spike = np.zeros(v_mp.shape, dtype=bool)
                input_spike[v_mp > threshold] = True
                v_mp -= (v_mp > threshold) * threshold

                self.v_mps[i] = v_mp
                self.spikes[i] = spike
                self.x_ks[i] = x_k

            self.a_is[-1] = np.concatenate([SpikingNetwork._calc_x_k(unit_spike, t) if len(unit_spike) > 0 else np.zeros((self.num_neurons[-1], 1))
                                            for unit_spike in self.spikes[-1]], axis=1)
            #if len(self.spikes[-1]) != 0:
            #    self.a_is[-1] = np.concatenate([SpikingNetwork._calc_x_k(unit_spike, t) for unit_spike in self.spikes[-1]], axis=1)
            if not np.any(input_spike):
                continue
            for unit_spike in self.spikes[-1]:
                unit_spike.append((t, input_spike[:, 0][:, np.newaxis]))
        self.t = t

        for i, _ in enumerate(self.v_mps):
            self.v_mps[i] = np.zeros(self.v_mps[i].shape)

    def backward(self, y):
        self.x_ks = [np.concatenate([SpikingNetwork._calc_x_k(unit_spike, self.t)
                                     if len(unit_spike) != 0 else np.ones((num_neuron, 1)) * 10 for unit_spike in spike], axis=1)
                     for spike, num_neuron in zip(self.spikes[:-1], self.num_neurons[:-1])]
        self.a_is = [np.concatenate([SpikingNetwork._calc_x_k(unit_spike, self.t)
                                     if len(unit_spike) != 0 else np.ones((num_neuron, 1)) * 10 for unit_spike in spike], axis=1)
                     for spike, num_neuron in zip(self.spikes[1:], self.num_neurons[1:])]

        sharp_spikes = self._calculate_sharp_spikes()
        has_spike_in_output = sharp_spikes.max(axis=0) > 0.0000001
        o_is = np.zeros(sharp_spikes.shape)
        o_is[:, has_spike_in_output] = sharp_spikes[:, has_spike_in_output] / sharp_spikes[:, has_spike_in_output].max(axis=0)
        delta = o_is - y

        def get_m_ls(spike):
            spike_t = list(filter(lambda x: x[0] == self.t, spike))
            total = spike_t[0][1].sum() if len(spike_t) != 0 else 0.0001
            if total < 0.00001:
                total = 0.0001
            return total
        m_ls = [np.array([get_m_ls(spike) for spike in unit_spike]) for unit_spike in self.spikes[:-1]]

        for i, weight, threshold, x_k, a_i, m_l in zip(
                reversed(range(len(m_ls))),
                reversed(self.weights),
                reversed(self.thresholds),
                reversed(self.x_ks), reversed(self.a_is),
                reversed(m_ls)):
            N_l = weight.shape[0]
            M_l = weight.shape[1]

            delta[:, ~has_spike_in_output] *= -1
            delta_weight = -SpikingNetwork.ETA_W * np.sqrt(N_l / m_l) * delta @ x_k.T
            delta_threshold = -SpikingNetwork.ETA_TH * np.sqrt(N_l / (m_l * M_l)) * delta * a_i

            if i - 1 >= 0:
                delta = (1 / self.thresholds[i - 1]) / np.sqrt((1 / m_ls[i - 1]) * np.sum((1 / self.thresholds[i - 1]) ** 2)) * np.sqrt(
                    M_l / m_l) * (weight.T @ delta)

            weight += delta_weight / delta.shape[1] - 0.0001 * weight

            '''
            weight_regularization = np.exp(SpikingNetwork.BETA * (np.sum(weight ** 2, axis=1) - 1))[:, np.newaxis]
            weight_regularization = SpikingNetwork.BETA * SpikingNetwork.LAMBDA * weight * np.concatenate([
            #weight_regularization = 0.5 * SpikingNetwork.LAMBDA * np.concatenate([
                weight_regularization for _ in range(weight.shape[1])], axis=1)
            #print('weight_regularization: {}'.format(weight_regularization))
            weight -= weight_regularization
            '''
            #weight += delta_weight - weight_regularization
            threshold += delta_threshold.mean(axis=1)[:, np.newaxis]

            '''
            weight_regularization = np.exp(SpikingNetwork.BETA * (np.sum(weight ** 2, axis=1) - 1))[:, np.newaxis]
            weight_regularization = SpikingNetwork.BETA * SpikingNetwork.LAMBDA * weight * np.concatenate([
            #weight_regularization = 0.5 * SpikingNetwork.LAMBDA * np.concatenate([
                weight_regularization for _ in range(weight.shape[1])], axis=1)
            #print('weight_regularization: {}'.format(weight_regularization))
            weight -= weight_regularization
            '''
            #weight += delta_weight - weight_regularization
            #threshold += delta_threshold

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
        max_sharp_spike = np.max(sharp_spikes, axis=0)
        return np.exp(sharp_spikes - max_sharp_spike) / np.sum(np.exp(sharp_spikes - max_sharp_spike), axis=0)

    @classmethod
    def _calc_x_k(cls, spike, t):
        return np.array([np.exp([(t_p - t) / cls.TAU_MP if fire else 0 for fire in value])
                        for t_p, value in spike]).sum(axis=0)[:, np.newaxis]

    def _calculate_sharp_spikes(self):
        return np.concatenate([np.array([spike[1] for spike in unit_spike]).sum(axis=0) if len(unit_spike) > 0 else np.zeros((self.num_neurons[-1], 1)) for unit_spike in self.spikes[-1]], axis=1)

    '''
    def _reset_weights(self):
        for i, num_neuron in enumerate(self.num_neurons):
            if i == len(self.weights):
                break

            root_3_per_m = np.sqrt(3 / num_neuron)
            #self.weights[i] = np.random.uniform(-root_3_per_m, root_3_per_m, self.weights[i].shape)
            self.weights[i] = np.random.uniform(0, root_3_per_m, self.weights[i].shape)

    '''
