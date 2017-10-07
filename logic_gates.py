from neural_network import NeuralNetwork
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__(2, 1)
        self.new_layer = torch.DoubleTensor([-15, 10, 10])
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
    def __call__(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(AND, self).forward(torch.DoubleTensor([x, y]))
        return output[0][0] > 0.5

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__(2, 1)
        self.new_layer = torch.DoubleTensor([-15, 30, 30])
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
    def __call__(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(OR, self).forward(torch.DoubleTensor([x, y]))
        return output[0][0] > 0.5

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__(1, 1)
        self.new_layer = torch.DoubleTensor([10, -20])
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
    def __call__(self, x):
        x = 1 if x else 0
        output = super(NOT, self).forward(torch.DoubleTensor([x]))
        return output[0][0] > 0.5

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__(2, 2, 1)
        self.new_layer = torch.DoubleTensor([[-40, 60, -60], [25, 50, -50]])
        layer = self.getLayer(1)
        for i in range(len(self.new_layer)):
            for j in range(len(self.new_layer[0])):
                layer[i][j] = self.new_layer[i][j]
        self.new_layer = torch.DoubleTensor([40, 80, -80])
        layer = self.getLayer(2)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
    def __call__(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(XOR, self).forward(torch.DoubleTensor([x, y]))
        return output[0][0] > 0.5
