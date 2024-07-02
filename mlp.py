import numpy as np
import random
import json
from abc import ABC, abstractmethod
import sys

import utility


class Cost(ABC):
    @staticmethod
    @abstractmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        pass

    @staticmethod
    @abstractmethod
    def error(z, a, y):
        """Return the error delta from the output layer."""
        pass


class QuadraticCost(Cost):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def error(z, a, y):
        return (a-y) * utility.sigmoid_prime(z)


class CrossEntropyCost(Cost):
    @staticmethod
    def fn(a, y):
        """
        Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def error(z, a, y):
        return (a-y)


class NeuralNetwork(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Input:
            sizes: The sizes of the neuron network. It is a list of 
            integers that describes the number of neurons contained 
            in each layer. The first layer (0th element) is input layer.

            cost: The cost function we choose to minimize for. The 
            available cost functions are: QuadraticCost, CrossEntropyCost
        
        Properties:
            The biases and weights for the network are initialized 
            randomly, using a Gaussian distribution with mean 0, and 
            variance 1.

            biases: A list of np array to represent biases of each neuron.
            biases[l][j][1] contains the value of bias in the (l+1)th layer.
            Note that the Network initialization code assumes that the 
            first layer of neurons is an input layer, and omits to set any 
            biases for those neurons.

            weights: A list of np array that to represent the weight 
            matrix of each neuron. weights[l][j][k] is the weight for the 
            connection between the kth neuron in the lth layer,
            and the jth neuron in the (l+1)th layer.

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(j, 1) for j in self.sizes[1:]]
        self.weights = [np.random.randn(j, k)/np.sqrt(k) for j, k in zip(self.sizes[1:], self.sizes[:-1])]

    def feedForward(self, a):
        """
        Input:
            a: the activation vector in the input layer.

        Output:
            return the activation vector in the output layer.
        """
        for b, w in zip(self.biases, self.weights): # for each layer from l=1 to l=L
            a = utility.sigmoid(np.dot(w, a) + b)
        return a

    def stochasticGradientDescend(self, training_data, epochs, mini_batch_size, training_rate,
                                   lmbda = 0.0,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_evaluation_accuracy=False,
                                   monitor_training_cost=False,
                                   monitor_training_accuracy=False):
        """
        Input:
            training_data: The "training_data" is a list of tuples
            "(x, y)" representing the training inputs and the desired
            outputs.

            epochs:

            mini_batch_size:

            training rate:

            test_data:



        """
        if evaluation_data:
            n_eval = len(evaluation_data) # number of sets of evaluation data
        n = len(training_data) # number of sets of training data 

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs): # repeat for (epochs) times
            random.shuffle(training_data) # The shuffle() method takes a sequence, like a list, and reorganize the order of the items.
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMinibatch(mini_batch, training_rate, lmbda, n)
            print("Epoch " +str(j)+" training complete")
            
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: "+str(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: "+str(accuracy)+" / "+str(n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: "+str(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: "+str(accuracy)+" / "+str(n_eval))
        
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def updateMinibatch(self, mini_batch, eta, lmbda, n): 
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        
        Input:
            mini_batch: a list of tuples "(x, y)"
            eta: the training rate
        
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backPropogation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Applying update rules
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]


    def backPropogation(self, x, y):
        """
        Input:
            x: training inputs, or activation
            y: desired output

        Output:
            Return a tuple "(nabla_b, nabla_w)" representing the
            gradient for the cost function C_x.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z=np.dot(w, activation)+b
            zs.append(z)
            activation=utility.sigmoid(z)
            activations.append(activation)
        
        # output error
        delta = self.cost.error(zs[-1], activations[-1], y)

        # backward pass
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = utility.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedForward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedForward(x)
            if convert: y = utility.vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def costDerivative(self, output_activation, y):
        return (output_activation-y)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    #### Loading a Network
    @staticmethod
    def load(filename):
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network.

        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        net = NeuralNetwork(data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net





