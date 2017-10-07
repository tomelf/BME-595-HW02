# BME-595 Assignment 02

## Part A

### Class NeuralNetwork
- Constructor: take the numbers of layers as input, randomly initialize a series of theta layers (including bias layer)
- getLayer: take one index of theta layers as input, return the specific theta layer
- forward: take a 1D tensor [DoubleTensor size k] or a 2D tensor [DoubleTensor size k x n] as input, forward the input to all layers and return the output

## Part B

### Class AND, OR, NOT, XOR

- Each class extends the class NeuralNetwork
- The following are the manually-crafted weights created by trial and error
  - AND: NeuralNetwork(2,1). First layer: [-15, 10, 10]
  - OR: NeuralNetwork(2,1). First layer: [-15, 30, 30]
  - NOT: NeuralNetwork(1,1). First layer: [10, -20]
  - XOR: NeuralNetwork(2,2,1). First layer: [[-40, 60, -60], [25, 50, -50]], Second layer: [40, 80, -80]
