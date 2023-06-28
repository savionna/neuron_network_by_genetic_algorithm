import numpy as np
import Population
import Person
import copy
import pickle

def read_data(train_file, test_file):
    X = []
    y = []
    with open(train_file, 'r') as train_f:
        train_lines = train_f.readlines()
        for line in train_lines:
            pattern, label = line.strip().split()
            X.append([int(bit) for bit in pattern])
            y.append(int(label))

    X_train = np.array(X)
    y_train = np.array(y)

    X_test = []
    Y_test = []
    with open(test_file, 'r') as test_f:
        test_lines = test_f.readlines()
        for line in test_lines:
            pattern2, label2 = line.strip().split()
            X_test.append([int(bit) for bit in pattern2])
            Y_test.append(int(label2))

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_train, y_train, X_test, Y_test

def main():
    # Read data
    train_file = 'train_file1.txt'
    test_file = 'test_file1.txt'
    X_train, y_train, X_test, test_real_lables = read_data(train_file, test_file)

    #check the person class:
    neural_net = Population.Population(X_train, y_train)
    neural_net.create_population()
    best_neural_net = copy.deepcopy(neural_net.next_generation())
    print("Test accurecy: ", best_neural_net.test(X_test, test_real_lables))

    # Its important to use binary mode
    dbfile = open('wnet1.txt', 'ab')
      
    # source, destination
    pickle.dump(best_neural_net, dbfile)                     
    dbfile.close()

if __name__ == "__main__":
    main()
