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
        
        while True:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))
            if converged:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        
        self.hidden_layers = 1
        self.hidden_nodes_by_layer = 50
        self.learning_rate = 0.1
        self.batch_size = 10
        self.error = 0.02

        self.layers = []
        self.biaises = []

        for _ in range(self.hidden_layers):
            self.layers.append(nn.Parameter(self.hidden_nodes_by_layer, self.hidden_nodes_by_layer))
            self.biaises.append(nn.Parameter(1, self.hidden_nodes_by_layer))

        if self.hidden_layers > 0:
            self.layers.insert(0, nn.Parameter(1, self.hidden_nodes_by_layer))
            self.layers.append(nn.Parameter(self.hidden_nodes_by_layer, 1))
        else:
            self.layers.insert(0, nn.Parameter(1, 1))
            self.layers.append(nn.Parameter(1, 1))

        self.biaises.append(nn.Parameter(1, 1))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        output = x
        print("-"*10)
        for i in range(self.hidden_layers+2):
            print("layer", i)
            output = nn.ReLU(output)
            output = nn.Linear(output, self.layers[i])
            output = nn.AddBias(output, self.biaises[i])
            print(output)

            
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
            gradients = nn.gradients(loss, self.layers)
            for i in range(self.hidden_layers+2):
                self.layers[i].update(gradients[i], -self.learning_rate)
            if nn.as_scalar(self.get_loss(x, y)) < self.error:
                break




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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
