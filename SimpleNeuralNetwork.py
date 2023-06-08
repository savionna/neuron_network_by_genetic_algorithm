import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.hidden_activation = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_activation = self.sigmoid(np.dot(self.hidden_activation, self.weights2) + self.bias2)
        return self.output_activation
    
    def backward(self, X, y, learning_rate):
        output_error = y - self.output_activation
        output_delta = output_error * self.sigmoid_derivative(self.output_activation)
        
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_activation)
        
        self.weights2 += np.dot(self.hidden_activation.T, output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights1 += np.dot(X.T, hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)