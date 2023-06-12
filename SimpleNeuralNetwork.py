import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, feature_number, hidden_size, output_size):
        self.feature_number = feature_number
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.5

        #initialize the parameters:
        self.W1 = np.random.randn(self.feature_number, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 2)
        self.b2 = np.zeros((1, 2))

    def train(self, X, y, num_of_epochs):
        train_size = len(X)

        for epoch in range(num_of_epochs):
            avg_epoch_loss = 0
            for i in range(train_size):
            # TODO :  Forward propagation

                z1 = np.dot(X[i], self.W1) + self.b1
                h1 = self.sigmoid(z1)
                Z2 = np.dot(h1, self.W2) + self.b2
                y_hat = self.softmax(Z2)
                y_true = np.zeros((1, 2))
                y_true[:, int(y[i])] = 1

                # TODO: Compute loss
                loss =  self.nll_loss(y_hat, y_true)
                avg_epoch_loss = avg_epoch_loss + loss



                # TODO: Back propagation - compute the gradients of each parameter
                dZ2 = (y_hat - y_true)
                dW2 = np.dot(dZ2.reshape(2, 1), h1)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dh1 = np.dot(dZ2, self.W2.T)
                dz1 = dh1 * self.sigmoid_derivative(z1)
                dW1 = np.dot(X[i].reshape(16, 1), dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # TODO: Update weights
                self.W2 = self.W2 - self.learning_rate * dW2.T
                self.b2 = self.b2 - self.learning_rate * db2
                self.W1 = self.W1 - self.learning_rate * dW1
                self.b1 = self.b1 - self.learning_rate * db1

            avg_epoch_loss = (avg_epoch_loss/train_size) 
            print("Epoch:", epoch," Loss:", avg_epoch_loss)
        
    #Negative Log Likelihood loss function for the multiclass
    def nll_loss(self, y_pred, y):
        loss = -np.sum(y * np.log(y_pred))
        return loss / float(y_pred.shape[0])
    #softmax function
    def softmax(self,z):
        return np.exp(z) / np.sum(np.exp(z))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def test(self,X, y):
        true_pred_counter = 0
        for i in range(len(X)):
            z1 = np.dot(X[i], self.W1) + self.b1
            h1 = self.sigmoid(z1)
            Z2 = np.dot(h1, self.W2) + self.b2
            y_hat = self.softmax(Z2)
            if np.argmax(y_hat) == int(y[i]):
                #print("predicted: " + str(np.argmax(y_hat)) + "real lable: " + str(int(y[i])))
                true_pred_counter += 1
        accuracy = true_pred_counter / len(X)

        return accuracy