# BME-595 Assignment 02

## Part A

1. Create a Python script and create an object model of class NeuralNetwork.
2. Initializing class using __init__(), with the list (in, h1, h2, …, out) as argument, will populate the network dictionary with the Θ(layer) matrices (which are mapping layer layer to layer + 1), initialised to random values (0 mean, 1/sqrt(layer_size) standard deviation). The size of the input layer is in, the size of the hidden layers are h1, h2, …, and the size of the output layer is out.
3. getLayer(layer) will return Θ(layer).
4. By running forward(input) the script will perform the forward propagation pass on the network previously built using sigmoid nonlinearities.


### Class NeuralNetwork
- Constructor: take the numbers of layers as input, randomly initialize a series of theta layers (including bias layer)
- getLayer: take one index of theta layers as input, return the specific theta layer
- forward: take a 1D tensor [DoubleTensor size k] or a 2D tensor [DoubleTensor size k x n] as input, forward the input to all layers and return the output

## Part B

1. Use the API in NeuralNetwork to create an AND, OR, NOT and XOR networks that perform logic operation on boolean values.
2. logic_gates.py will have four classes as per the second API. Each class constructor will call NeuralNetwork class and then set the weights of neural network using getLayer([int] layer).
3. Calling forward function of any logic operation will call forward() of NeuralNetwork and return the output of the logic operation.

### Class AND, OR, NOT, XOR

- Each class extends the class NeuralNetwork
- The following are the manually-crafted weights created by trial and error
  - AND: NeuralNetwork(2,1). First layer: [-15, 10, 10]
  - OR: NeuralNetwork(2,1). First layer: [-15, 30, 30]
  - NOT: NeuralNetwork(1,1). First layer: [10, -20]
  - XOR: NeuralNetwork(2,2,1). First layer: [[-40, 60, -60], [25, 50, -50]], Second layer: [40, 80, -80]
