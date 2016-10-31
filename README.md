# Simple Neural Network for MNIST Digit Recognition.

Total of three layers:
    1. Input.
    2. Hidden.
    3. Output.


## Input Layer
- nIn = 784 units + 1. 
- Connected to hidden by V.shape = (nHid, nIn + 1)


## Hidden Layer
- 200 units + 1. 
- Connected to output by W.shape = (nOut, nHid + 1)


## Output Layer
- 10 units.
- Softmax activation function.

## Training
- Cross-entropy (logistic) loss function.
- SGD for weight updates.