# rust_nn

A Neural Network library written in Rust.

Test 1
Training on MNIST data set with following configuration:
1. Two hidden layers with 50 activation functions (RELU)
2. Softmax classifier
3. Minibatch size 200
4. Learning rate 0.05
5. No regularization
6. Minibatch optimizer

![train-test curve](https://github.com/mariusdanciu/rust_nn/blob/master/train.png)

Test 2
Training on MNIST data set with following configuration:
1. Two hidden layers with 50 activation functions (RELU)
2. Softmax classifier
3. Minibatch size 200
4. Learning rate 0.01
5. No regularization
6. Adam optimizer

![train-test curve](https://github.com/mariusdanciu/rust_nn/blob/master/train_adam.png)

**Test results
1 With Adam optimizer we see ~95% accuracy on ~360 iterations
2. Without Adam we see ~ 93% accuracy after 1200 iteration.

**Thus with Adam optimizer we observe a reach to a slightly higher accuracy ~3.4 times faster.

The library supports:
1. L2 regularization
2. Minibatch, Momentum, RMSProp and Adam optimizations
3. Only Dense layers so far.
4. No GPU acceleration yet
