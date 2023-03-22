The code to train an Artificial Neural Network to recognise
handwritten digits using backpropagation is explained in this
section. The ANN we are about to use is a two-layer ANN, with a
hidden layer and an output layer apart from the input layer.


To train the ANN, we require datasets with the input dataset and
their corresponding outputs as a CSV file. We need to download the
file before proceeding further. After downloading, we import and
store the file in a variable called ‘data’ in the form of an array.

Our ANN will have a simple two-layer architecture. The input layer
will have 784 units corresponding to the 784 pixels in each 28x28
input image. A hidden layer will have 10 units with ReLU activation,
and finally, our output layer a[2] will have 10 units corresponding to
the ten digits with softmax activation.
Z[1] = W[1]X + b[1]
A[1] = gReLU (Z[1])
Z[2] = W[2]A[1] + b[2]
A[2] = gsoftmax (Z[2])

If the values from the output layer do not match with the desired
value, we need to propagate backward and update the weights and
biases in order to minimize the error. ‘one_hot(Y)’ represents the
desired outputs. The formula for change in weights and biases is as
follows:
Backward propagation

dZ[2] = A[2] − Y
dW[2] = (1/m) * dZ[2] * A[1]T
dB[2] = (1/m) * ΣdZ[2]
dZ[1] = W[2]T dZ[2]. ∗ g[1]′ (z[1])
dW[1] = (1/m) dZ[1] A[0]T
dB[1] = (1/m) ΣdZ[1]


Parameter updates

W[2] = W[2] − αdW[2]
b[2] = b[2] − αdb[2]
W[1] = W[1] − αdW[1]
b[1] = b[1] − αdb[1]


❖ ‘get_predictions’ function returns the index corresponding to the
maximum value in the output layer, i.e. highest probability.
❖ ‘get_accuracy’ cross-checks the training data output values and
the actual output values and gives out the ratio of total correct
predictions.
❖ ‘gradient_descent’ puts together all the functions mentioned
earlier and creates the ANN model.


The next step is to run our ANN and train it using the training
dataset. The 500 represents the number of iterations, i.e., the
number of cycles the ANN should undergo.


