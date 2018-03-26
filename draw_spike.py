# coding: utf-8

import time
import pygame
from pygame.locals import QUIT


class DrawSpike:
    WIDTH = 700
    HEIGHT = 600
    RADIUS = 5
    COL_MAX_NEURONS = 50

    def __init__(self, network):
        pygame.init()
        self.screen = pygame.display.set_mode((DrawSpike.WIDTH, DrawSpike.HEIGHT))
        pygame.display.set_caption('Spiking MoYF')
        self.network = network

    def update(self):
        self.screen.fill((255, 255, 255))
        num_layers = len(self.network.num_neurons)
        #max_num_neurons = max(self.network.num_neurons)

        #radius = min(DrawSpike.WIDTH / num_layers * 0.75, DrawSpike.HEIGHT / max_num_neurons * 0.85) * 0.5

        for i, num_neuron in enumerate(self.network.num_neurons):
            base_x = DrawSpike.WIDTH / (num_layers + 1) * (i + 1)
            for j in range(num_neuron):
                x = base_x + j // DrawSpike.COL_MAX_NEURONS * DrawSpike.RADIUS
                y = DrawSpike.HEIGHT / ((num_neuron + 1) if (num_neuron + 1) < DrawSpike.COL_MAX_NEURONS else DrawSpike.COL_MAX_NEURONS) * \
                    ((j + 1) % DrawSpike.COL_MAX_NEURONS)

                '''
                if i < num_layers - 1:
                    next_x = DrawSpike.WIDTH / (num_layers + 1) * (i + 2)
                    for k in range(self.network.num_neurons[i + 1]):
                        next_y = DrawSpike.HEIGHT / (num_neuron + 1) * (k + 1)
                        pygame.draw.line(self.screen, (0, 0, 0), (x, y), (next_x, next_y), 2)
                '''

                #import bpdb; bpdb.set_trace()
                #pygame.draw.ellipse(self.screen, (0, 0, 0), (x - radius, y - radius, radius * 2, radius * 2), 2)
                args = [
                    self.screen,
                    (0, 0, 0),
                    (x - DrawSpike.RADIUS, y - DrawSpike.RADIUS, DrawSpike.RADIUS * 2, DrawSpike.RADIUS * 2)
                ]
                pygame.draw.ellipse(*args)
                if self.network.spikes[i][-1][j][0]:
                    args[1] = (255, 0, 0)
                    pygame.draw.ellipse(*args)
                    #pygame.draw.ellipse(self.screen, (255, 0, 0), (x - radius + 1, y - radius + 1, radius * 2 - 2, radius * 2 - 2))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        time.sleep(1)
