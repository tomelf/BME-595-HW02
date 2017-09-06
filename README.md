# BME-595 Assignment 02

## Part A

### Class NeuralNetwork
- Constructor: take the numbers of layers as input, randomly initialize a series of theta layers (including bias layer)
- getLayer: take one index of theta layers as input, return the specific theta layer
- forward: take a 1D tensor [DoubleTensor size k] or a 2D tensor [DoubleTensor size n x k] as input, forward the input to all layers and return the output

## Part B

### Class AND, OR, NOT, XOR

- Each class extends the class NeuralNetwork
- The first step is coverting x, y from boolean to (1,0), then perform the following operations:
  - AND: the logic is to get the boolean value of [torch.add(x,y).ge(2) == 1]
  - OR: the logic is to get the boolean value of [torch.add(x,y).ge(1) == 1]
  - NOT: the logic is to get the boolean value of [~x == 1]
  - XOR: the logic is to get the boolean value of [torch.add(x*~y, ~x*y).ge(1) == 1]
