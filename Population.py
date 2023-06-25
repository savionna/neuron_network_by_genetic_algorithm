import numpy as np
import random
import Person
import math
import copy


class Population:
    def __init__(self, x_train , y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.population = []
        self.layers_amount = 3
        self.size_of_layers = [16, 4, 2]
        #self.size_of_layers = [16, 32, 128, 32, 16, 2]
        self.population_size = 200
        self.mutation_rate = 0.9
        self.generation_number = 400
        self.feature_number = x_train.shape[1]
        

    def create_population(self):
        # Initialize the population
        for _ in range(self.population_size):
            person = Person.Person(self.x_train, self.y_train, self.layers_amount, self.size_of_layers, self.feature_number)
            self.population.append(person)

    def get_population(self):
        return self.population
    def get_population_size(self):
        return self.population_size
            

    def next_generation(self):
        mutation_precent = int(self.population_size*self.mutation_rate)
        for generation in range(self.generation_number):
            offspring = []
            fitness_scores = self.compute_fitness()

            # print("taco")
            # for person in self.population:
            #     print(person.get_fitness())
            max_index = np.argmax(fitness_scores)
            #print("The best fit: ", fitness_scores[max_index])
            best_person = self.population[max_index]
            #print("Best person fit: ", best_person.get_fitness())

            amount = best_person.get_fitness()*10
            amount = math.ceil((self.population_size/100)*amount)

            for i in range(amount):
                offspring.append(Person.Person(self.x_train, self.y_train, self.layers_amount, self.size_of_layers, self.feature_number, copy.deepcopy(best_person.get_weights())))
            
            # Select parents for mating
            parents = self.selection(fitness_scores)
            # Generate offspring through crossover and mutation
            people_after_cross = self.crossover(parents)

            #select people for mutataion:
            random_indexes = random.sample(range(len(people_after_cross)), mutation_precent)
            for index in random_indexes:
                people_after_cross[index].mutate()

            for y in range(self.population_size - amount):
                offspring.append(copy.deepcopy(people_after_cross[y]))


            # Replace the old population with the offspring
            self.population = offspring

            # Print the best fitness score in the current generation
            best_fitness = np.max(fitness_scores)
            print("Generation:", generation, "Best Fitness:", best_fitness)

    def compute_fitness(self):
        fitness_scores = []
        for person in self.population:
            person.train()
            fitness_scores.append(person.get_fitness())
        return np.array(fitness_scores)

    def selection(self, fitness_scores):
        # Calculate the number of individuals to select (70% of the population size)
        num_to_select = int(0.2 * self.population_size)

        # Sort the fitness scores in descending order
        sorted_indices = np.argsort(fitness_scores)[::-1]

        # Select the indices corresponding to the top individuals
        selected_indices = sorted_indices[:num_to_select]

        # Calculate the probabilities for selection
        selected_scores = fitness_scores[selected_indices]
        probabilities = selected_scores / np.sum(selected_scores)
        # Roulette wheel selection
        # Randomly select parents based on the probabilities
        parents_indices = np.random.choice(selected_indices, size=self.population_size, replace=True, p=probabilities)
        parents = [self.population[i] for i in parents_indices]
        # print(fitness_scores)
        # for p in parents:
        #     print(p.get_fitness())
        return parents

    def crossover(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]

            random_place = random.randint(0, len(parent1.weights) - 1)
            shape = parent1.weights[random_place].shape
            random_indices = tuple(np.random.randint(0, dim) for dim in shape)

            w1 = [parent1.weights[j] for j in range(random_place)]
            w2 = [parent2.weights[j] for j in range(random_place)]

            # Create copies of the input arrays to avoid modifying them directly
            array1_crossover = parent1.weights[random_place].copy()
            array2_crossover = parent2.weights[random_place].copy()

            # Perform crossover from the random indices till the end of the arrays
            array1_crossover[random_indices[0], random_indices[1]:] = parent2.weights[random_place][random_indices[0], random_indices[1]:]
            array2_crossover[random_indices[0], random_indices[1]:] = parent1.weights[random_place][random_indices[0], random_indices[1]:]
            array1_crossover[random_indices[0]+1:, :] = parent2.weights[random_place][random_indices[0]+1:, :]
            array2_crossover[random_indices[0]+1:, :] = parent1.weights[random_place][random_indices[0]+1:, :]

            w1.append(array1_crossover)
            w2.append(array2_crossover)

            if random_place < len(parent1.weights) - 1:
                w1.extend(parent2.weights[j] for j in range(random_place + 1, len(parent1.weights)))
                w2.extend(parent1.weights[j] for j in range(random_place + 1, len(parent1.weights)))

            offspring1 = copy.deepcopy(parent1)
            offspring2 = copy.deepcopy(parent2)
            offspring1.weights, offspring2.weights = w1, w2

            offspring.extend([offspring1, offspring2])

        return offspring
        
    def test(self,x, y):
        pass