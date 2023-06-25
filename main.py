import numpy as np
import Population
import Person

def read_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_train = int(len(lines) * 3 / 4)  # 3/4 of rows for training
        train_lines = lines[:num_train]
        test_lines = lines[num_train:]

        for line in train_lines:
            pattern, label = line.strip().split()
            X.append([int(bit) for bit in pattern])
            y.append(int(label))

    X_train = np.array(X)
    y_train = np.array(y)

    X_test = []
    Y_test = []
    for line in test_lines:
        pattern2, label2 = line.strip().split()
        X_test.append([int(bit) for bit in pattern2])
        Y_test.append(int(label2))

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    return X_train, y_train, X_test, Y_test

def main():
    # Read data
    file = 'nn0.txt'
    X_train, y_train, X_test, test_real_lables = read_data(file)
    feature_number = X_train.shape[1]
    
    # Create and train the neural network
    #neural_net = Population.Population(X_train, y_train)
    #accuracy = neural_net.test(X_test, test_real_lables)
    #print(accuracy)

    #check the person class:
    neural_net = Population.Population(X_train, y_train)
    neural_net.create_population()
    # for person in neural_net.get_population():
    #     person.train()
    neural_net.next_generation()
    



    #check the person class:
    # size_of_layers = [16, 32, 128, 32, 16, 2]
    # print(feature_number)
    # test_person = Person.Person(X_train, y_train, 6, size_of_layers, feature_number)
    # test_person.train()

if __name__ == "__main__":
    main()
