import torch
import numpy as np

class NeuralNetwork(object):
    def __init__(self, in_layer, *h_arr):
        self.layers = []
        self.in_layer = in_layer
        self.out_layer = h_arr[-1]
        for i in range(len(h_arr)):
            s = in_layer if i==0 else h_arr[i-1]
            e = h_arr[i]
            l = np.random.normal(0, 1/np.sqrt(e), (s+1,e))
            self.layers.append(torch.DoubleTensor(l))

    def getLayer(self, layer):
        return self.layers[layer-1]

    def forward(self, input):
        output = input.view(1, input.size()[0]) if len(input.size()) == 1 else input
        for idx, layer in enumerate(self.layers):
            bias = torch.randn(output.size()[0], 1).type(torch.DoubleTensor)
            output = torch.cat((bias, output), 1)
            output = torch.sigmoid(output.mm(layer))
        return output
