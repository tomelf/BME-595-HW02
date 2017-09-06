from neural_network import NeuralNetwork as NN
from logic_gates import AND, OR, NOT, XOR
import torch

def main():
    nn = NN(10, 10, 5, 2)
    input = torch.randn(3, 10).type(torch.DoubleTensor)
    print("input", input)
    output = nn.forward(input)
    print("output", output)

    # And = AND()
    # Or = OR()
    # Not = NOT()
    # Xor = XOR()
    #
    # print(And(True, True))
    # print(And(True, False))
    # print(And(False, True))
    # print(And(False, False))
    # print(Or(True, True))
    # print(Or(True, False))
    # print(Or(False, True))
    # print(Or(False, False))
    # print(Not(True))
    # print(Not(False))
    # print(Xor(True, True))
    # print(Xor(True, False))
    # print(Xor(False, True))
    # print(Xor(False, False))

if __name__ == "__main__":
    main()
