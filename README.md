# rust_nn

A Neural Network library written in Rust. Training on MNIST data set with following configuration:
1. Two hidden layers with 50 activation functions (RELU)
2. Softmax classifier
3. Minibatch size 200
4. Learning rate 0.05


After 2 epochs ~ 93 % accuracy using 1000 examples from the test data-set. The train-test accuracy curve is this :

![train-test curve](https://github.com/mariusdanciu/rust_nn/blob/master/train.png)

The library supports:
1. L2 regularization
2. Momentum optimizer (RMSProp and Adam will follow)
3. Only Dense layers so far.
