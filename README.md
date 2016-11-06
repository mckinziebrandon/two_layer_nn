# Simple Neural Network for MNIST Digit Recognition.

## Evaluation:

* __Accuracy__: classifies the Kaggle MNIST test set (10k samples) with 98.2 percent accuracy. 
    - 99.97 percent training accuracy (50k training samples).
    - 97.53 percent validation accuracy (10k validation samples). 
* __Runtime__: With 784 feature (input) units, 1200 hidden units, and 10 outputs units (all layers fully connected), the network
trained by using 60k samples (training+validation) in minibatches of 50 each, and 15 full passes (epochs) over the the training
data. On my laptop, this took __six minutes and 21 seconds__ for the most recent session. For 800 units, __six minutes and 8
seconds__. 

Total of three layers:
    1. Input.
        - nIn = 784 units + 1. 
        - Connected to hidden by V.shape = (nHid, nIn + 1)
    2. Hidden.
        - RELU activation function.
        - 1200 units + 1. 
        - Connected to output by W.shape = (nOut, nHid + 1)
    3. Output.
        - 10 units.
        - Softmax activation function.

## Training
- Cross-entropy (logistic) loss function.
- (Mini)Batch Gradient Descent via backpropagation for weight updates.

## Figures
![Plots][TrainingLoss]



[TrainingLoss]: png_dir/LossAndAcc1.png
