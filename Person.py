import numpy as np
import random

class Person:
    def __init__(self, x_train , y_train, layers_amount, size_of_layers, feature_number, weights=None):
        self.x_train = x_train
        self.y_train = y_train
        self.layers_amount = layers_amount
        self.size_of_layers = size_of_layers
        self.feature_number = feature_number
        self.weights = weights
        self.fitness = 0

        if weights is None:
            self.init_Weights()

    def init_Weights(self):
        self.weights = []
        feature_size = self.feature_number
        for i in range(self.layers_amount):
            self.weights.append(np.random.randn(feature_size, self.size_of_layers[i]))
            feature_size = self.size_of_layers[i]
    
    def get_weights(self):
        return self.weights

    
    def softmax(self,z):
        return np.exp(z) / np.sum(np.exp(z))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def train(self):
        h1 = self.x_train.reshape(self.x_train.shape[0], 1, self.feature_number)
        for layer in range(self.layers_amount):
            z = np.dot(h1, self.weights[layer])
            h1 = self.sigmoid(z)
        # Flatten the output array
        h1 = h1.reshape(h1.shape[0], -1)
        # Calculate y_hat for all samples
        y_hat = np.argmax(self.softmax(h1), axis=1)

        # Calculate accuracy
        self.fitness = np.mean(y_hat == self.y_train)
        #print(self.accuracy)
    
    def get_fitness(self):
        return self.fitness
    
    def mutate(self):
        random_number = random.randint(0, len(self.weights) - 1)
        weight_matrix = self.weights[random_number]

        # Generate random indices within the weight matrix shape
        random_indices = tuple(np.random.randint(0, dim) for dim in weight_matrix.shape)

        # Perform mutation by adding a random value from a normal distribution to the selected element
        mutated_value = weight_matrix[random_indices] + np.random.randn()

        # Create a copy of the individual and update the mutated weight value
        self.weights[random_number][random_indices] = mutated_value
    
    def test(self, x, y):
        true_pred_counter = 0
        for i in range(len(x)):
            a = x[i]
            for layer in range(self.layers_amount):
                z = np.dot(a, self.weights[layer])
                a = self.sigmoid(z)

            y_hat = float(np.argmax(self.softmax(a)))
            y_true = float(y[i])

            if y_hat == y_true:
                true_pred_counter += 1

        accuracy = true_pred_counter / len(y)
        return accuracy
    
    
    


    
