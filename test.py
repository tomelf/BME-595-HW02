from neural_network import NeuralNetwork as NN
from logic_gates import AND, OR, NOT, XOR
import torch

def main():
    # nn = NN(2, 1)
    # input = torch.DoubleTensor([[1,1,0,0],[1,0,1,0]])
    # print("input", input)
    # output = nn.forward(input)
    # print("output", output)

    And = AND()
    Or = OR()
    Not = NOT()
    Xor = XOR()
    
    print(And(True, True), And(True, False), And(False, True), And(False, False))
    print(Or(True, True), Or(True, False), Or(False, True), Or(False, False))
    print(Not(True), Not(False))
    print(Xor(True, True), Xor(True, False), Xor(False, True), Xor(False, False))

if __name__ == "__main__":
    main()
