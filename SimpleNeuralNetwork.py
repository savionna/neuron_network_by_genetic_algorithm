import numpy as np
import random

# class SimpleNeuralNetwork:
#     def __init__(self, feature_number, hidden_size, output_size):
#         self.feature_number = feature_number
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = 0.5

#         #initialize the parameters:
#         self.W1 = np.random.randn(self.feature_number, self.hidden_size)
#         self.b1 = np.zeros((1, self.hidden_size))
#         self.W2 = np.random.randn(self.hidden_size, 2)
#         self.b2 = np.zeros((1, 2))

#     def train(self, X, y, num_of_epochs):
#         train_size = len(X)

#         for epoch in range(num_of_epochs):
#             avg_epoch_loss = 0
#             for i in range(train_size):
#             # TODO :  Forward propagation

#                 z1 = np.dot(X[i], self.W1) + self.b1
#                 h1 = self.sigmoid(z1)
#                 Z2 = np.dot(h1, self.W2) + self.b2
#                 y_hat = self.softmax(Z2)
#                 y_true = np.zeros((1, 2))
#                 y_true[:, int(y[i])] = 1

#                 # TODO: Compute loss
#                 loss =  self.nll_loss(y_hat, y_true)
#                 avg_epoch_loss = avg_epoch_loss + loss



#                 # TODO: Back propagation - compute the gradients of each parameter
#                 dZ2 = (y_hat - y_true)
#                 dW2 = np.dot(dZ2.reshape(2, 1), h1)
#                 db2 = np.sum(dZ2, axis=0, keepdims=True)

#                 dh1 = np.dot(dZ2, self.W2.T)
#                 dz1 = dh1 * self.sigmoid_derivative(z1)
#                 dW1 = np.dot(X[i].reshape(16, 1), dz1)
#                 db1 = np.sum(dz1, axis=0, keepdims=True)

#                 # TODO: Update weights
#                 self.W2 = self.W2 - self.learning_rate * dW2.T
#                 self.b2 = self.b2 - self.learning_rate * db2
#                 self.W1 = self.W1 - self.learning_rate * dW1
#                 self.b1 = self.b1 - self.learning_rate * db1

#             avg_epoch_loss = (avg_epoch_loss/train_size) 
#             print("Epoch:", epoch," Loss:", avg_epoch_loss)

class SimpleNeuralNetwork:
    def __init__(self, feature_number, hidden_size, output_size):
        self.feature_number = feature_number
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_size = 10
        self.mutation_rate = 0.1
        self.num_generations = 10
        self.W1 = np.random.standard_normal((self.feature_number, self.hidden_size))
        self.W2 = np.random.standard_normal((self.hidden_size, self.output_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.hidden_size))
        # Initialize the population
        self.population = []
        for _ in range(self.population_size):
            chromosome = {}
            # chromosome['W1'] = np.random.randn(self.feature_number, self.hidden_size)
            # chromosome['b1'] = np.zeros((1, self.hidden_size))
            # chromosome['W2'] = np.random.randn(self.hidden_size, self.output_size)
            # chromosome['b2'] = np.zeros((1, self.output_size))
            chromosome['W1'] = np.random.standard_normal((self.feature_number, self.hidden_size))
            chromosome['b1'] = np.zeros((1, self.hidden_size))
            chromosome['W2'] = np.random.standard_normal((self.hidden_size, self.output_size))
            chromosome['b2'] = np.zeros((1, self.output_size))
            self.population.append(chromosome)




    def train(self, X, y):
        for generation in range(self.num_generations):
            fitness_scores = self.compute_fitness(X, y)

            # Select parents for mating
            parents = self.selection(fitness_scores)

            # Generate offspring through crossover and mutation
            offspring = self.crossover(parents)
            #offspring = self.mutation(offspring)

            # Replace the old population with the offspring
            self.population = offspring

            # Print the best fitness score in the current generation
            best_fitness = np.max(fitness_scores)
            print("Generation:", generation, "Best Fitness:", best_fitness)

        # Select the best chromosome after all generations
        best_chromosome = self.population[np.argmax(fitness_scores)]

        # Set the weights of the neural network to the best chromosome
        self.W1 = best_chromosome['W1']
        self.b1 = best_chromosome['b1']
        self.W2 = best_chromosome['W2']
        self.b2 = best_chromosome['b2']
        # Update the weights of the neural network based on the best chromosome
        self.update_weights()

    def update_weights(self):
        self.weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
    }

    def compute_fitness(self, X, y):
        fitness_scores = []
        for chromosome in self.population:
            W1 = chromosome['W1']
            b1 = chromosome['b1']
            W2 = chromosome['W2']
            b2 = chromosome['b2']

            accuracy = self.calculate_accuracy(X, y, W1, W2, b1, b2)
            fitness_scores.append(accuracy)

        return np.array(fitness_scores)

    def selection(self, fitness_scores):
        # Roulette wheel selection
        probabilities = fitness_scores / np.sum(fitness_scores)
        parents_indices = np.random.choice(len(self.population), size=self.population_size, replace=True, p=probabilities)
        parents = [self.population[i] for i in parents_indices]

        return parents

    def crossover(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Perform crossover at a random point
            crossover_point = random.randint(0, self.feature_number * self.hidden_size + self.hidden_size * self.output_size)
            child1 = {}
            child2 = {}

            for key in parent1.keys():
                parent1_genes = parent1[key].flatten()
                parent2_genes = parent2[key].flatten()

                child1_genes = np.concatenate((parent1_genes[:crossover_point], parent2_genes[crossover_point:]))
                child2_genes = np.concatenate((parent2_genes[:crossover_point], parent1_genes[crossover_point:]))

                child1[key] = child1_genes.reshape(parent1[key].shape)
                child2[key] = child2_genes.reshape(parent2[key].shape)

            offspring.append(child1)
            offspring.append(child2)

        return offspring

    def mutation(self, offspring):
        for chromosome in offspring:
            for key in chromosome.keys():
                genes = chromosome[key].flatten()

                for i in range(len(genes)):
                    if random.random() < self.mutation_rate:
                        # Apply mutation by randomly perturbing the gene
                        genes[i] += np.random.normal(0, 0.1)

                chromosome[key] = genes.reshape(chromosome[key].shape)

        return offspring

    # Remaining code for the test, sigmoid, softmax functions, etc.

        
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
    
    def calculate_accuracy(self,X, y, w1, w2, b1, b2):
        true_pred_counter = 0
        for i in range(len(X)):
            z1 = np.dot(X[i], self.W1)
            #print(X[i].shape)
            print("w1: " + str(self.W1.shape))
            print("w2: " + str(self.W2.shape))
            print("z1: " + str(z1.shape))
            h1 = self.sigmoid(z1)
            print("h1: " + str(h1.shape))
            z2 = np.dot(h1, self.W2)
            print("z2: " + str(z2.shape))
            y_hat = self.sigmoid(z2)
            print(y_hat.shape)
            predicted_labels = np.where(y_hat >= 0.5, 1, 0)
            if int(predicted_labels) == int(y[i]):
                true_pred_counter += 1
        accuracy = true_pred_counter / len(X)
        #print(accuracy)
        return accuracy


 #     if np.argmax(y_hat) == int(y[i]):
        #         #print("predicted: " + str(np.argmax(y_hat)) + "real lable: " + str(int(y[i])))
        #         true_pred_counter += 1
        # accuracy = true_pred_counter / len(X)
        # #print("accuracy: " + str(accuracy))
        # return accuracy
        # Round the predicted value to either 0 or 1
# Round the predicted value to either 0 or 1

    def test(self,X, y):

        # Use the updated weights for testing
        self.W1 = self.weights['W1']
        self.b1 = self.weights['b1']
        self.W2 = self.weights['W2']
        self.b2 = self.weights['b2']
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