"""Implementation of Stochastic gradient descent learning algorithm for feedforward neural network. Gradients are
    calculated using Backpropagation."""

import random
import numpy as np
import mnist_loader

class Network(object):
    def __init__(self,sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes # contains the number of neurons in the respective layer of the network in [2,3,1] format.
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        """Return the output of the network if the input is 'a' """
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        """Train the network with mini batch stochastic gradient descent"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
            for k in xrange(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data,n_test))
            else:
                print "Epoch {0}: complete".format(j)

    def update_mini_batch(self,mini_batch,eta):
        """Update the weights and biases by applying gradient descent using backpropagation to a single mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_b_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backdrop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch)*nw) for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch) * nb) for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        """Return a tuple representing the gradient for the cost function"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = [] #store all the z vectors,layer by layer
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in xrange(2,self.num_layer):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(),delta)*sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta,activations[-1-1].transpose())
        return (nabla_b,nabla_w)

    def evaluate(self,test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_result = [(np.argmax(self.feedforward(x)),y)
                       for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_result)
    def cost_derivative(self,output_activations,y):
        """Return the partial derivative for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))



