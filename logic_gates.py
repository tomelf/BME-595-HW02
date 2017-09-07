from neural_network import NeuralNetwork
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__(3, 1)
        self.new_layer = torch.DoubleTensor([-100, 30, 40, 30])
    def __call__(self, x, y):
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(2), torch.ByteTensor([1]))

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__(3, 1)
        self.new_layer = torch.DoubleTensor([-100, 30, 40, 30])
    def __call__(self, x, y):
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(1), torch.ByteTensor([1]))

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__(3, 1)
        self.new_layer = torch.DoubleTensor([-100, 30, 40, 30])
    def __call__(self, x):
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        return torch.equal(~b1, torch.ByteTensor([1]))

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__(3, 1)
        self.new_layer = torch.DoubleTensor([-100, 30, 40, 30])
    def __call__(self, x, y):
        layer = self.getLayer(1)
        for i, v in enumerate(self.new_layer):
            layer[0][i] = self.new_layer[i]
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1*~b2, ~b1*b2).ge(1), torch.ByteTensor([1]))
