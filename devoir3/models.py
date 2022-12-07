# Group Members :
#
# BRANLY Stéphane (MATRICULE 2232279)
# GUICHARD Amaury (MATRICULE 2227083)
#


import math
import time
import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """

        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        

        self.learning_rate = 0.1
        self.batch_size = 50
        self.error = 0.01

        self.layers = []
        self.biaises = []
        self.layer_sizes = [1,200,200,200,1]

        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Parameter(self.layer_sizes[i], self.layer_sizes[i+1]))
            self.biaises.append(nn.Parameter(1, self.layer_sizes[i+1]))



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        output = x
        for i in range(len(self.layers)-1):
            output = nn.Linear(output, self.layers[i])
            output = nn.AddBias(output, self.biaises[i])
            output = nn.ReLU(output)
        i+=1
        output = nn.Linear(output, self.layers[i])
        output = nn.AddBias(output, self.biaises[i])
            
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)

            if nn.as_scalar(loss) < self.error:
                break

            gradients = nn.gradients(loss, self.layers + self.biaises)
            for i in range(len(self.layers)):
                self.layers[i].update(gradients[i], -self.learning_rate)
                self.biaises[i].update(gradients[i+len(self.layers)], -self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        
        self.learning_rate = 0.5
        self.batch_size = 500
        self.error = 0.0265
        self.layer_sizes = [784, 250,125,50, 10]

        self.layers = []
        self.decay = 0.1
            
        self.biaises = []

        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Parameter(self.layer_sizes[i], self.layer_sizes[i+1]))
            self.biaises.append(nn.Parameter(1, self.layer_sizes[i+1]))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        
        output = x
        for i in range(len(self.layers)-1):
            output = nn.Linear(output, self.layers[i])
            output = nn.AddBias(output, self.biaises[i])
            output = nn.ReLU(output)
        i+=1
        output = nn.Linear(output, self.layers[i])
        output = nn.AddBias(output, self.biaises[i])
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        ite = 0
        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)

            if dataset.get_validation_accuracy() > 1 - self.error:
                break

            # le learning rate decroit au fur et a mesure des itérations
            # idée de https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
            lrate = self.learning_rate / (1 + self.decay * ite/800)
            gradients = nn.gradients(loss, self.layers + self.biaises)
            for i in range(len(self.layers)):
                self.layers[i].update(gradients[i], -lrate)
                self.biaises[i].update(gradients[i+len(self.layers)], -lrate)
            ite +=1
            
